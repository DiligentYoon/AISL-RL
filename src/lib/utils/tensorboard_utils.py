from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_PREFIXES = ["Episode", "Reward", "Task Penalty", "Task Reward"]

# -----------------------------
# Util functions
# -----------------------------
def _read_scalar_tags_from_event_file(event_file: str | Path) -> List[str]:
    """ Read only 'tag' (not value loaded)."""
    event_file = Path(event_file)
    if not event_file.exists():
        raise FileNotFoundError(event_file)

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    return ea.Tags().get("scalars", [])


def common_scalar_tags(event_files: List[str | Path]) -> List[str]:
    """
    Return sorted list of scalar tags that exist in ALL event files.
    """
    if len(event_files) == 0:
        return []

    tag_sets = []
    for f in event_files:
        tags = _read_scalar_tags_from_event_file(f)
        tag_sets.append(set(tags))

    common = set.intersection(*tag_sets) if tag_sets else set()
    return sorted(common)

# -----------------------------
# 1) Event file -> scalar DF
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
    Read multiple event files and concatenate them.
    Adds column: source_file
    If only_common_tags=True, keeps only tags that exist in ALL files.
    """
    if len(event_files) == 0:
        return pd.DataFrame(columns=["source_file", "tag", "step", "wall_time", "value"])

    common = None
    if only_common_tags:
        common = set(common_scalar_tags(event_files))
        if len(common) == 0:
            # 공통 태그가 없으면 빈 DF 반환
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


def plot_prefix_figures_from_event_files(
    event_files: List[str | Path],
    *,
    legend_titles: Optional[List[str]] = None,
    task_name: str = "Task",                         
    prefixes: List[str] = DEFAULT_PREFIXES,
    only_common_tags: bool = True,
    x_axis: str = "step",
    smoothing: int = 1,
    max_cols: int = 2,
    figsize_per_subplot: Tuple[float, float] = (10.0, 3.0),
    save_dir: Optional[str | Path] = None,
    save_prefix: str = "tb",
    show: bool = True,
    include_unmatched: bool = False,
    case_sensitive: bool = False,
) -> Dict[str, Path]:
    if x_axis not in ("step", "wall_time"):
        raise ValueError("x_axis must be 'step' or 'wall_time'")

    if legend_titles is not None and len(legend_titles) != len(event_files):
        raise ValueError(
            f"legend_titles length ({len(legend_titles)}) must match event_files length ({len(event_files)})."
        )

    df = read_scalars_from_event_files(event_files, only_common_tags=only_common_tags)
    if df.empty:
        raise ValueError(
            "No data to plot. "
            "Either there are no scalar events, or (only_common_tags=True) yields empty intersection."
        )

    all_tags = sorted(df["tag"].unique().tolist())
    groups = group_tags_by_prefix(
        all_tags, prefixes=prefixes, case_sensitive=case_sensitive,
        allow_unmatched=True, unmatched_key="Unmatched"
    )

    # source_file -> legend title
    file_strs = [str(Path(f)) for f in event_files]
    if legend_titles is None:
        legend_titles = [Path(f).name for f in file_strs]
    file_to_title = {file_strs[i]: legend_titles[i] for i in range(len(file_strs))}

    # save setup
    saved: Dict[str, Path] = {}
    save_dir_path = None
    if save_dir is not None:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

    def _plot_one_group(group_name: str, tags: List[str]):
        if not tags:
            return

        n = len(tags)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))
        fig_w = figsize_per_subplot[0] * ncols
        fig_h = figsize_per_subplot[1] * nrows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        fig.suptitle(
            f"{task_name} | {group_name} ({n} common tags)" if only_common_tags else f"{task_name} | {group_name} ({n} tags)",
            y=1.02
        )

        ax_list = axes.ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

        for ax, tag in zip(ax_list, tags):
            g_tag = df[df["tag"] == tag]

            for src, g_src in g_tag.groupby("source_file"):
                g_src = g_src.copy()

                if x_axis == "step":
                    g_src = _dedup_by_step(g_src).sort_values("step")
                    x = g_src["step"].to_numpy()
                else:
                    g_src = g_src.drop_duplicates(subset=["wall_time"], keep="last").sort_values("wall_time")
                    x = g_src["wall_time"].to_numpy()

                y = _moving_average(g_src["value"].to_numpy(), smoothing)
                label = file_to_title.get(src, Path(src).name)
                ax.plot(x, y, label=label)

            ax.set_title(f"{task_name} | {tag}", fontsize=10)

            ax.grid()
            ax.set_xlabel(x_axis)
            ax.set_ylabel("value")
            ax.legend(fontsize=8)

        for ax in ax_list[len(tags):]:
            ax.axis("off")

        fig.tight_layout()

        if save_dir_path is not None:
            safe_group = group_name.replace("/", "__").replace(" ", "_").replace(":", "_")
            safe_task = task_name.replace("/", "__").replace(" ", "_").replace(":", "_")
            out_path = save_dir_path / f"{save_prefix}__{safe_task}__{safe_group}.png"
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

# ======================================================================================================

task_name = "G1_recovery"

event_files = [
    "logs/g1_recovery/2026-03-13_00-05-45_mappo/events.out.tfevents.1773328004.AIS-simulation.17508.0",
    "logs/g1_recovery/2026-03-13_12-24-10_mappo/events.out.tfevents.1773372308.AIS-simulation.24328.0"
]

legends = ["Cooperative MAPPO", "Vanilla MAPPO"]

# Plot
plot_prefix_figures_from_event_files(
    event_files,
    legend_titles=legends,
    task_name=task_name,
    only_common_tags=True,
    smoothing=11,
    max_cols=2,
    show=True,
)
