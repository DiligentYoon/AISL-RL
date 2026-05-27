"""
Stage 2 of the SafeFall baseline pipeline.

Train the GRU fall predictor offline on a dataset produced by
``collect_offline.py``. Reads ``meta.json`` (including a snapshot of the
safe_fall cfg block) and the .pt shards under
``<dataset_dir>/shards/``; writes checkpoints / TensorBoard / metrics under

This script intentionally avoids Isaac Lab — it only needs PyTorch.
"""

import argparse
import collections
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Repo root: this file is .../src/main/reach_avoid/baselines/safefall/train_offline.py
# Add src to sys.path so `lib.*` imports resolve when running as a module from elsewhere.
# import sys
# _THIS = os.path.abspath(os.path.dirname(__file__))
# _SRC_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
# if _SRC_ROOT not in sys.path:
#     sys.path.insert(0, _SRC_ROOT)


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Offline training for the SafeFall GRU predictor.")
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to a dataset directory produced by collect_offline.py.")
parser.add_argument("--seed", type=int, default=None, help="Override seed from cfg snapshot.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from lib.buffer.baselines.recurrent_replay import RecurrentReplayBuffer
from lib.model.Baselines.SafeFall.safe_fall import GRU

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Offline training for the SafeFall GRU predictor.")
#     parser.add_argument("--dataset_dir", type=str, required=True,
#                         help="Path to a dataset directory produced by collect_offline.py.")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
#                         help="Compute device.")
#     parser.add_argument("--seed", type=int, default=None, help="Override seed from cfg snapshot.")
#     return parser.parse_args()


def _load_dataset_meta(dataset_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(dataset_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_buffer(meta: Dict[str, Any], sf_cfg: Dict[str, Any], device: torch.device,
                  split: str, dataset_dir: str) -> RecurrentReplayBuffer:
    buf = RecurrentReplayBuffer(
        buffer_size=1,                       # Reallocated inside load_from_disk.
        num_envs=1,
        device=device,
        seq_len=sf_cfg["buffer"]["seq_len"],
        fall_lead_seconds=sf_cfg["buffer"]["fall_lead_seconds"],
        step_dt=meta["step_dt"],
        max_episode_steps=meta["max_episode_steps"],
    )
    buf.init_buffer(obs_dim=meta["obs_dim"])
    buf.load_from_disk(dataset_dir, split=split)
    return buf


@torch.no_grad()
def evaluate(model: nn.Module,
             val_buf: RecurrentReplayBuffer,
             threshold: float,
             batch_size: int,
             seq_len: int,
             step_dt: float) -> Dict[str, float]:
    """Compute accuracy / false-alarm / mean lead-time on the validation buffer."""
    model.eval()

    n_val_eps = len(val_buf.closed_episodes)
    if n_val_eps == 0:
        return {"val_loss": float("nan"), "val_acc": float("nan"),
                "val_false_alarm": float("nan"), "val_lead_time_s": float("nan")}

    # 1) Loss / accuracy / false alarm over fixed-length sequences.
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    n_safe_tokens = 0
    n_safe_false_pos = 0

    num_batches = max(1, math.ceil(n_val_eps / batch_size))
    for _ in range(num_batches):
        obs, label, mask = val_buf.sample_sequences(batch_size=batch_size, seq_len=seq_len)
        logits, _, _ = model(obs)
        flat_logits = logits.reshape(-1, 2)
        flat_label = label.reshape(-1)
        flat_mask = mask.reshape(-1)

        ce = F.cross_entropy(flat_logits, flat_label, reduction="none")
        m_f = flat_mask.float()
        denom = m_f.sum().clamp(min=1.0)
        total_loss += float((ce * m_f).sum() / denom)

        preds = flat_logits.argmax(dim=-1)
        valid = flat_mask
        total_correct += int(((preds == flat_label) & valid).sum())
        total_valid += int(valid.sum())

        safe_mask = valid & (flat_label == 0)
        n_safe_tokens += int(safe_mask.sum())
        n_safe_false_pos += int((safe_mask & (preds == 1)).sum())

    # 2) Lead time on terminated episodes (full sequence per episode).
    lead_times_steps = []
    for ep in val_buf.closed_episodes:
        if not ep.get("terminated", False):
            continue
        rows_t = torch.as_tensor(ep["rows"], device=val_buf.device, dtype=torch.long)
        obs_seq = val_buf.tensors["obs"][rows_t, 0, :].unsqueeze(0)   # (1, T_e, obs_dim)
        logits, _, _ = model(obs_seq)
        probs = logits.softmax(dim=-1)[0, :, 1]                         # (T_e,)
        triggered = (probs > threshold).nonzero(as_tuple=False)
        if triggered.numel() == 0:
            continue
        first_trigger = int(triggered[0].item())
        lead_steps = ep["length"] - first_trigger
        lead_times_steps.append(lead_steps)

    lead_time_s = (float(np.mean(lead_times_steps)) * step_dt) if lead_times_steps else float("nan")

    metrics = {
        "val_loss":         total_loss / max(num_batches, 1),
        "val_acc":          total_correct / max(total_valid, 1),
        "val_false_alarm":  n_safe_false_pos / max(n_safe_tokens, 1),
        "val_lead_time_s":  lead_time_s,
    }
    return metrics


def main():
    args = args_cli
    device = torch.device(args.device)

    # ---- Load dataset metadata (includes a snapshot of safe_fall cfg). ----
    dataset_dir = os.path.abspath(args.dataset_dir)
    meta = _load_dataset_meta(dataset_dir)
    if "safe_fall_cfg" not in meta:
        raise KeyError(
            f"meta.json at {dataset_dir} is missing 'safe_fall_cfg'. "
            "Regenerate the dataset with the updated collect_offline.py."
        )
    sf_cfg = meta["safe_fall_cfg"]
    tr_cfg = sf_cfg["train"]
    eval_cfg = sf_cfg["eval"]
    model_cfg = sf_cfg["model"]

    seed = args.seed if args.seed is not None else meta.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Buffers (load happens here; tensors are reallocated to packed layout). ----
    print(f"[INFO] Loading train split from {dataset_dir} ...")
    train_buf = _build_buffer(meta, sf_cfg, device, split="train", dataset_dir=dataset_dir)
    print(f"[INFO] Loaded {len(train_buf.closed_episodes)} train trajectories "
          f"({train_buf.buffer_size} packed steps).")

    print(f"[INFO] Loading val split ...")
    val_buf = _build_buffer(meta, sf_cfg, device, split="val", dataset_dir=dataset_dir)
    print(f"[INFO] Loaded {len(val_buf.closed_episodes)} val trajectories.")

    # ---- Model + optimizer. ----
    model = GRU(
        obs_dim=meta["obs_dim"],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )
    grad_norm_clip = float(tr_cfg.get("grad_norm_clip", 0.0))
    batch_size = int(tr_cfg["batch_size"])
    seq_len = int(sf_cfg["buffer"]["seq_len"])
    threshold = float(eval_cfg.get("threshold", 0.5))
    # epochs = int(tr_cfg["learning_epochs"])
    epochs = 20
    checkpoint_interval = int(epochs / 5)

    # ---- Run dir. ----
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(dataset_dir, f"run_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dump effective config + dataset meta link.
    with open(os.path.join(run_dir, "training_cfg.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_dir": dataset_dir,
            "safe_fall_cfg": sf_cfg,
            "seed": seed,
        }, f, indent=2)

    n_train_eps = len(train_buf.closed_episodes)
    steps_per_epoch = max(1, math.ceil(n_train_eps / batch_size))

    print("=" * 72)
    print(f"  epochs       : {epochs}")
    print(f"  batch_size   : {batch_size}")
    print(f"  seq_len      : {seq_len}")
    print(f"  steps/epoch  : {steps_per_epoch}")
    print(f"  lr / wd      : {tr_cfg['learning_rate']} / {tr_cfg.get('weight_decay', 0.0)}")
    print(f"  threshold    : {threshold}")
    print(f"  run_dir      : {run_dir}")
    print("=" * 72)

    history = []
    best_score = -float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_valid = 0

        for step in range(steps_per_epoch):
            obs, label, mask = train_buf.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            logits, _, _ = model(obs)

            flat_logits = logits.reshape(-1, 2)
            flat_label = label.reshape(-1)
            flat_mask = mask.reshape(-1).float()
            ce = F.cross_entropy(flat_logits, flat_label, reduction="none")
            loss = (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

            optim.zero_grad()
            loss.backward()
            if grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optim.step()

            with torch.no_grad():
                preds = flat_logits.argmax(dim=-1)
                valid = flat_mask.bool()
                epoch_correct += int(((preds == flat_label) & valid).sum())
                epoch_valid += int(valid.sum())
                epoch_loss += float(loss)

        train_loss = epoch_loss / steps_per_epoch
        train_acc = epoch_correct / max(epoch_valid, 1)

        metrics = evaluate(
            model, val_buf,
            threshold=threshold,
            batch_size=batch_size,
            seq_len=seq_len,
            step_dt=meta["step_dt"],
        )

        elapsed = time.time() - start_time
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **metrics,
            "elapsed_s": elapsed,
        })

        print(f"[epoch {epoch+1:02d}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f} "
              f"FA={metrics['val_false_alarm']:.4f} "
              f"lead={metrics['val_lead_time_s']:.3f}s "
              f"({elapsed:.1f}s)")

        # Per-epoch checkpoint (mirrors train.py: <predictor>_agent_<step>.pt naming style).
        if (epoch+1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"safefall_agent_{epoch+1}.pt")
            torch.save({"critic": model.state_dict()}, ckpt_path)

        # Best-by metric: higher lead time at low false alarm is better; tie-break on accuracy.
        score = -metrics["val_false_alarm"] + (metrics["val_lead_time_s"] if not math.isnan(metrics["val_lead_time_s"]) else 0.0)
        if score > best_score:
            best_score = score
            torch.save({"critic": model.state_dict()}, os.path.join(ckpt_dir, "best.pt"))

    with open(os.path.join(run_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 72)
    print(f"  Final checkpoint : {os.path.join(ckpt_dir, 'best.pt')}")
    print(f"  Run directory    : {run_dir}")
    print("=" * 72)

    simulation_app.close()

if __name__ == "__main__":
    main()
