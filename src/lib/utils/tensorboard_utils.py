from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_PREFIXES = ["Episode", "Reward", "Task Penalty", "Task Reward"]


# -----------------------------
# Tag / scalar reading
# -----------------------------
def _read_scalar_tags_from_event_file(event_file: str | Path) -> List[str]:
    """Read only 'tag' (no value loading)."""
    event_file = Path(event_file)
    if not event_file.exists():
        raise FileNotFoundError(event_file)

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    return ea.Tags().get("scalars", [])


def common_scalar_tags(event_files: List[str | Path]) -> List[str]:
    """Return sorted list of scalar tags present in ALL event files."""
    if len(event_files) == 0:
        return []

    tag_sets = [set(_read_scalar_tags_from_event_file(f)) for f in event_files]
    common = set.intersection(*tag_sets) if tag_sets else set()
    return sorted(common)


# -----------------------------
# Event file -> scalar DF
# -----------------------------
def read_scalars_from_event_file(event_file: str | Path) -> pd.DataFrame:
    event_file = Path(event_file)
    if not event_file.exists():
        raise FileNotFoundError(event_file)

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    rows = []
    for tag in tags:
        for ev in ea.Scalars(tag):
            rows.append(
                {"tag": tag, "step": int(ev.step), "wall_time": float(ev.wall_time), "value": float(ev.value)}
            )
    return pd.DataFrame(rows)


def read_scalars_from_event_files(
    event_files: List[str | Path],
    *,
    only_common_tags: bool = True,
) -> pd.DataFrame:
    """
    Read multiple event files and concatenate them. Adds a 'source_file'
    column. With only_common_tags=True, keeps only tags that exist in ALL
    files (intersection).
    """
    if len(event_files) == 0:
        return pd.DataFrame(columns=["source_file", "tag", "step", "wall_time", "value"])

    common = None
    if only_common_tags:
        common = set(common_scalar_tags(event_files))
        if len(common) == 0:
            return pd.DataFrame(columns=["source_file", "tag", "step", "wall_time", "value"])

    dfs = []
    for f in event_files:
        df = read_scalars_from_event_file(f)
        if df.empty:
            continue
        df["source_file"] = str(Path(f))
        if common is not None:
            df = df[df["tag"].isin(common)]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["source_file", "tag", "step", "wall_time", "value"])

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["tag", "source_file", "step", "wall_time"]).reset_index(drop=True)
    return out


# -----------------------------
# Helpers
# -----------------------------
def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def _dedup_by_step(g: pd.DataFrame) -> pd.DataFrame:
    return g.sort_values(["step", "wall_time"]).drop_duplicates(subset=["step"], keep="last")


def group_tags_by_prefix(
    tags: List[str],
    prefixes: List[str] = DEFAULT_PREFIXES,
    *,
    case_sensitive: bool = False,
    allow_unmatched: bool = True,
    unmatched_key: str = "Unmatched",
) -> Dict[str, List[str]]:
    if not case_sensitive:
        tags_cmp = [t.lower() for t in tags]
        prefixes_cmp = [p.lower() for p in prefixes]
    else:
        tags_cmp = tags
        prefixes_cmp = prefixes

    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]), reverse=True)

    groups: Dict[str, List[str]] = {p: [] for p in prefixes}
    if allow_unmatched:
        groups[unmatched_key] = []

    for tag, tag_c in zip(tags, tags_cmp):
        matched = None
        for i in order:
            p = prefixes[i]
            p_c = prefixes_cmp[i]
            if p_c in tag_c:
                matched = p
                break
        if matched is None:
            if allow_unmatched:
                groups[unmatched_key].append(tag)
        else:
            groups[matched].append(tag)

    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


# -----------------------------
# Event file discovery
# -----------------------------
def find_event_files(root: str | Path) -> List[Path]:
    """
    Recursively find tfevents files under ``root``. When a single run
    directory contains multiple event files, keep only the most recently
    modified one.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    candidates = sorted(root.rglob("events.out.tfevents.*"))
    by_run: Dict[Path, Path] = {}
    for f in candidates:
        run_dir = f.parent
        prev = by_run.get(run_dir)
        if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
            by_run[run_dir] = f
    return sorted(by_run.values())


# -----------------------------
# Cross-run aggregation
# -----------------------------
def aggregate_runs_by_tag(
    df: pd.DataFrame,
    *,
    x_axis: str = "step",
    num_points: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Per-tag aggregation across runs. Returns ``{tag: (x_grid, mean, std, n_runs)}``.

    For each tag, run sequences (x, y) are extracted (deduplicated, sorted).
    If every run shares the same x vector, statistics are computed without
    interpolation; otherwise a common grid spanning the intersection of run
    ranges is built and each run is linearly interpolated onto it. Std uses
    ddof=1 (falls back to zeros when only one run is available).
    """
    if x_axis not in ("step", "wall_time"):
        raise ValueError("x_axis must be 'step' or 'wall_time'")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    if df.empty:
        return out

    for tag, g_tag in df.groupby("tag"):
        runs: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, g_src in g_tag.groupby("source_file"):
            g_src = g_src.copy()
            if x_axis == "step":
                g_src = _dedup_by_step(g_src).sort_values("step")
                x = g_src["step"].to_numpy(dtype=float)
            else:
                g_src = g_src.drop_duplicates(subset=["wall_time"], keep="last").sort_values("wall_time")
                x = g_src["wall_time"].to_numpy(dtype=float)
            y = g_src["value"].to_numpy(dtype=float)
            if x.size == 0:
                continue
            runs.append((x, y))

        if not runs:
            continue

        same_grid = all(
            r[0].shape == runs[0][0].shape and np.array_equal(r[0], runs[0][0])
            for r in runs
        )

        if same_grid:
            x_grid = runs[0][0]
            stacked = np.stack([r[1] for r in runs], axis=0)
        else:
            x_min = max(r[0].min() for r in runs)
            x_max = min(r[0].max() for r in runs)
            if x_max <= x_min:
                # No overlapping range across runs; skip rather than fabricate.
                continue
            n_grid = num_points if num_points is not None else int(np.median([len(r[0]) for r in runs]))
            n_grid = max(2, n_grid)
            x_grid = np.linspace(x_min, x_max, n_grid)
            stacked = np.stack([np.interp(x_grid, r[0], r[1]) for r in runs], axis=0)

        mean = stacked.mean(axis=0)
        if stacked.shape[0] >= 2:
            std = stacked.std(axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)
        out[tag] = (x_grid, mean, std, stacked.shape[0])

    return out


# -----------------------------
# Plot (mean ± std)
# -----------------------------
def plot_prefix_statistics(
    event_files: List[str | Path],
    save_dir: str | Path,
    *,
    prefixes: List[str] = DEFAULT_PREFIXES,
    only_common_tags: bool = True,
    x_axis: str = "step",
    smoothing: int = 1,
    num_points: Optional[int] = None,
    max_cols: int = 2,
    figsize_per_subplot: Tuple[float, float] = (10.0, 3.0),
    save_prefix: str = "tb",
    show: bool = True,
    include_unmatched: bool = False,
    case_sensitive: bool = False,
    band_alpha: float = 0.2,
) -> Dict[str, Path]:
    if x_axis not in ("step", "wall_time"):
        raise ValueError("x_axis must be 'step' or 'wall_time'")
    if not event_files:
        raise ValueError("No event files were provided.")

    df = read_scalars_from_event_files(event_files, only_common_tags=only_common_tags)
    if df.empty:
        raise ValueError(
            "No data to plot. Either there are no scalar events, or "
            "(only_common_tags=True) yields an empty intersection."
        )

    stats = aggregate_runs_by_tag(df, x_axis=x_axis, num_points=num_points)
    if not stats:
        raise ValueError("Aggregation produced no tags. Check that runs share overlapping ranges.")

    all_tags = sorted(stats.keys())
    groups = group_tags_by_prefix(
        all_tags, prefixes=prefixes, case_sensitive=case_sensitive,
        allow_unmatched=True, unmatched_key="Unmatched",
    )

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    def _plot_one_group(group_name: str, tags: List[str]):
        if not tags:
            return

        n = len(tags)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))
        fig_w = figsize_per_subplot[0] * ncols
        fig_h = figsize_per_subplot[1] * nrows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        ax_list = axes.ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

        for ax, tag in zip(ax_list, tags):
            x_grid, mean, std, _n_runs = stats[tag]
            mean_s = _moving_average(mean, smoothing)
            std_s = _moving_average(std, smoothing)

            line, = ax.plot(x_grid, mean_s)
            ax.fill_between(
                x_grid, mean_s - std_s, mean_s + std_s,
                alpha=band_alpha, color=line.get_color(), linewidth=0,
            )

            ax.set_title(tag, fontsize=10)
            ax.grid()
            ax.set_xlabel(x_axis)
            ax.set_ylabel("value")

        for ax in ax_list[len(tags):]:
            ax.axis("off")

        fig.tight_layout()

        safe_group = group_name.replace("/", "__").replace(" ", "_").replace(":", "_")
        out_path = save_dir_path / f"{save_prefix}__{safe_group}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        saved[group_name] = out_path

        if show:
            plt.show()
        else:
            plt.close(fig)

    for p in prefixes:
        _plot_one_group(p, groups.get(p, []))

    if include_unmatched:
        _plot_one_group("Unmatched", groups.get("Unmatched", []))

    return saved


# -----------------------------
# CLI
# -----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mean ± std reward curves aggregated across multiple TensorBoard runs."
    )
    parser.add_argument("--log-dir", required=True,
                        help="Directory containing run subdirectories with events.out.tfevents.* files.")
    parser.add_argument("--smoothing", type=int, default=1,
                        help="Moving-average window applied to mean and std (>=1).")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    log_dir = Path(args.log_dir)
    event_files = find_event_files(log_dir)
    if not event_files:
        raise SystemExit(f"No event files found under {log_dir}")

    print(f"[INFO] Found {len(event_files)} event file(s) under {log_dir}:")
    for f in event_files:
        print(f"  - {f}")

    save_dir = log_dir / "figures"
    plot_prefix_statistics(
        event_files,
        save_dir=save_dir,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()
