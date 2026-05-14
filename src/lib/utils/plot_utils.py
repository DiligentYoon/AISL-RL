from __future__ import annotations

import math
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from abc import abstractmethod


# ============================================================
# Basic visualization helpers
# ============================================================
class Ball:
    def __init__(self, ax, size=10, shape="o"):
        self.ax = ax
        self.scatter, = self.ax.plot([], [], [], shape, markersize=size, animated=False)

    def update(self, pos):
        x, y, z = pos
        self.scatter.set_data_3d([x], [y], [z])


class Line:
    def __init__(self, ax, size=1, color="g"):
        self.ax = ax
        self.line, = self.ax.plot([], [], [], linewidth=size, color=color, animated=False)

    def update(self, pos):
        # pos: (3, N)
        self.line.set_xdata(pos[0, :])
        self.line.set_ydata(pos[1, :])
        self.line.set_3d_properties(np.asarray(pos[2, :]))


class CircleRegion2D:
    def __init__(
        self,
        ax,
        radius=0.1,
        center=(0.0, 0.0),
        edgecolor="c",
        facecolor="c",
        alpha=0.15,
        linewidth=2.0,
    ):
        self.ax = ax
        self.patch = Circle(
            center,
            radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        self.ax.add_patch(self.patch)

    def update(self, center, radius, visible=True):
        self.patch.center = (center[0], center[1])
        self.patch.radius = max(float(radius), 0.0)
        self.patch.set_visible(visible)

    def artist(self):
        return self.patch


class LIPM_3D_Animate:
    def __init__(self, ax):
        self.ax = ax

        self.origin = Ball(ax, size=2, shape="ko")
        self.COM_trajectory = Line(ax, size=1, color="g")
        self.COM_head = Ball(ax, size=2, shape="ro")

        self.left_foot = Ball(ax, size=5, shape="bo")
        self.right_foot = Ball(ax, size=5, shape="mo")

        self.left_leg = Line(ax, size=3, color="b")
        self.right_leg = Line(ax, size=3, color="m")
        self.COM = Ball(ax, size=16, shape="ro")

    def update(self, COM_pos, COM_pos_trajectory, left_foot_pos, right_foot_pos):
        self.origin.update([0.0, 0.0, 0.0])
        self.COM.update(COM_pos)
        self.COM_trajectory.update(COM_pos_trajectory)
        self.COM_head.update(COM_pos_trajectory[:, -1])

        self.left_foot.update(left_foot_pos)
        self.right_foot.update(right_foot_pos)

        pos_1 = np.zeros((3, 2))
        pos_1[:, 0] = COM_pos
        pos_1[:, 1] = left_foot_pos
        self.left_leg.update(pos_1)

        pos_2 = np.zeros((3, 2))
        pos_2[:, 0] = COM_pos
        pos_2[:, 1] = right_foot_pos
        self.right_leg.update(pos_2)

        # camera follow
        self.ax.set_xlim(COM_pos[0] - 2.0, COM_pos[0] + 8.0)
        self.ax.set_ylim(COM_pos[1] - 2.0, COM_pos[1] + 2.0)
        self.ax.set_zlim(-0.01, 1.0)

        return [
            self.origin.scatter,
            self.left_leg.line,
            self.right_leg.line,
            self.COM.scatter,
            self.COM_trajectory.line,
            self.COM_head.scatter,
            self.left_foot.scatter,
            self.right_foot.scatter,
        ]


# ============================================================
# Base GIF plotter (generic infrastructure)
# ============================================================
class GIFSavePlotter:
    """
    Abstract base class for offline GIF-saving plotters.

    Responsibilities
    ----------------
    - Data buffering: collects one step of viz_data per append() call,
      keyed by the entries in plot_cfg.
    - Episode tracking: records (start, end) frame indices per episode so
      that subclasses can slice trajectories within an episode boundary.
    - GIF export: drives FuncAnimation + PillowWriter using the subclass-
      supplied _setup_figure / _init_func / _update_func hooks.

    Subclass contract
    -----------------
    Implement exactly three methods:
      _setup_figure()      -- create self.fig and all axes / artist objects
      _init_func()         -- initialise artists to frame-0 state, return artist list
      _update_func(i: int) -- update artists to frame i state, return artist list

    Everything else (buffering, saving, episode math) is handled here.
    """

    def __init__(self, env, plot_cfg: dict, plot_dir):
        self.env = env
        self.cfg = plot_cfg
        self.dt = self.env._unwrapped.step_dt
        self.plot_dir = plot_dir if plot_dir is not None else os.getcwd()

        # Keyed buffer: one list per plot_cfg entry
        self.buffer: dict[str, list] = {key: [] for key in plot_cfg.keys()}

        # Episode boundary tracking
        self.episode_ranges: list[tuple[int, int]] = []
        self.current_episode_start: int = 0

        # Figure handle — populated by _setup_figure()
        self.fig = None

        self._setup_figure()

    # ----------------------------------------------------------
    # Data methods (shared across all subclasses)
    # ----------------------------------------------------------
    def process_data(self, x):
        """
        Convert incoming step data to a numpy array or Python scalar.

        Handles torch.Tensor, numpy arrays, and plain scalars.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 0:
            return x.item()
        return x.copy()

    def append(self, viz_data: dict, episode_end: bool = False):
        """
        Buffer one step of visualization data.

        Parameters
        ----------
        viz_data : dict
            Must contain every key present in plot_cfg.
        episode_end : bool
            Set True on the last frame of an episode so the episode
            boundary is recorded for trajectory slicing in _update_func.
        """
        for key in self.buffer.keys():
            if key not in viz_data:
                raise KeyError(f"Missing key in viz_data: '{key}'")
            self.buffer[key].append(self.process_data(viz_data[key]))

        curr_idx = self.num_frames() - 1
        if episode_end:
            self.episode_ranges.append((self.current_episode_start, curr_idx))
            self.current_episode_start = curr_idx + 1

    def reset(self):
        """Clear all buffered data and episode records."""
        for key in self.buffer:
            self.buffer[key].clear()
        self.episode_ranges.clear()
        self.current_episode_start = 0

    def num_frames(self) -> int:
        """Return the total number of buffered frames."""
        first_key = next(iter(self.buffer.keys()))
        return len(self.buffer[first_key])

    def _val(self, key: str, i: int):
        """Return the buffered value for key at frame index i."""
        return self.buffer[key][i]

    def _get_episode_range_for_frame(self, i: int) -> tuple[int, int]:
        """
        Return the (start, end) frame indices of the episode that contains
        frame i.  Raises IndexError if i is not covered by any range.
        """
        for start, end in self.episode_ranges:
            if start <= i <= end:
                return start, end
        raise IndexError(f"Frame index {i} is not covered by any episode range.")

    def _finalize_open_episode(self):
        """
        Close the current (still-open) episode so that all buffered frames
        are covered by an episode range before saving.
        """
        n = self.num_frames()
        if n == 0:
            return
        last_closed_end = -1 if len(self.episode_ranges) == 0 else self.episode_ranges[-1][1]
        if last_closed_end < n - 1:
            self.episode_ranges.append((self.current_episode_start, n - 1))
            self.current_episode_start = n

    def _to_2d_series(self, key: str) -> tuple[np.ndarray, list[str]]:
        """Convert buffered list for `key` into (n_frames, n_components) plus column names.

        - Scalars or length-1 arrays produce one column named `{key}`.
        - Length-N arrays (N>1) produce N columns named `{key}_0` ... `{key}_{N-1}`.
        - Higher-rank arrays are flattened via reshape(-1).
        - Steps with mismatched lengths are padded with NaN to the max width.
        """
        arrs = [np.asarray(v).reshape(-1) for v in self.buffer[key]]
        if len(arrs) == 0:
            return np.empty((0, 1), dtype=float), [key]
        width = max(a.size for a in arrs)
        out = np.full((len(arrs), width), np.nan, dtype=float)
        for i, a in enumerate(arrs):
            out[i, : a.size] = a
        cols = [key] if width == 1 else [f"{key}_{j}" for j in range(width)]
        return out, cols

    # ----------------------------------------------------------
    # GIF export (shared across all subclasses)
    # ----------------------------------------------------------
    def save(self, filename: str = "animation.gif"):
        """
        Render and save the buffered data as an animated GIF.

        Parameters
        ----------
        filename : str
            Output filename written inside self.plot_dir.
        """
        self._finalize_open_episode()
        data_len = self.num_frames()

        if data_len == 0:
            raise RuntimeError("No buffered data. Call append() first.")
        if data_len == 1:
            raise RuntimeError("Need at least 2 frames to build animation.")

        anim = FuncAnimation(
            fig=self.fig,
            init_func=self._init_func,
            func=self._update_func,
            frames=range(1, data_len),
            interval=1000 * self.dt,
            blit=False,
            repeat=False,
        )

        fps = max(1, int(1.0 / self.dt))
        writer = PillowWriter(fps=fps)

        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, filename)
        anim.save(filepath, writer=writer)
        print(f"--------- Saved animation to: {filepath}")

    def close(self):
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)

    # ----------------------------------------------------------
    # Abstract hooks (subclass must implement)
    # ----------------------------------------------------------
    @abstractmethod
    def _setup_figure(self):
        """Create self.fig and all axes / artist objects."""
        raise NotImplementedError

    @abstractmethod
    def _init_func(self) -> list:
        """Initialise all artists to their frame-0 state. Return artist list."""
        raise NotImplementedError

    @abstractmethod
    def _update_func(self, i: int) -> list:
        """Update all artists to frame i. Return artist list."""
        raise NotImplementedError


# ============================================================
# Capturability-specific plotter
# ============================================================
class CapturabilityPlotter(GIFSavePlotter):
    """
    GIF plotter for capturability visualization.

    Panels
    ------
    - Left  : 3-D walking animation (LIPM model)
    - Right top    : Bird-eye view with ICP and capture region
    - Right bottom : m_step(t) time-series

    Notes
    -----
    Panel options (axis limits, follow mode, etc.) are set as instance
    attributes before super().__init__() so that _setup_figure() can
    read them when it is called from the parent constructor.
    """

    REQUIRED_KEYS = (
        "com_pos",
        "left_foot_pos",
        "right_foot_pos",
        "icp_pos",
        "capture_region_center",
        "capture_region_radius",
        "time_hist",
        "risk_value",
        "icp_ankle_dist_hist",
    )

    def __init__(self, env, plot_cfg: dict, plot_dir=None):
        missing = [k for k in self.REQUIRED_KEYS if k not in plot_cfg]
        if missing:
            raise KeyError(f"plot_cfg missing required keys: {missing}")

        # Figure size
        self.figsize = (14, 10)

        # 3D panel options
        self.ax3d_xlim = (-2.0, 8.0)
        self.ax3d_ylim = (-2.0, 2.0)
        self.ax3d_zlim = (-0.01, 1.0)
        self.ax3d_box_aspect = (6.0, 4.0, 3.0)
        self.ax3d_view_init = (20, -150)

        # Bird-eye-view options
        self.bx_xlim = (-0.5, 5.0)
        self.bx_ylim = (-1.0, 1.0)
        self.bx_follow_com = True
        self.bx_follow_xspan = (-2.0, 4.0)
        self.bx_follow_yspan = (-2.0, 2.0)

        # m_step plot options
        self.mx_initial_xlim = (0.0, 3.0)
        self.mx_follow_window = 2.0
        self.mx_future_margin = 0.2

        # Annotation format string
        self.COM_pos_str = "COM = (%.2f, %.2f)\nICP-ankle = %.3f m\nradius = %.3f m\nm_step = %.3f m"

        # Matplotlib artist handles — populated by _setup_figure()
        self.ax = None
        self.bx = None
        self.mx = None
        self.LIPM_3D_ani = None
        self.capture_region_ani = None
        self.ani_text_COM_pos = None
        self.original_ani = None
        self.COM_traj_ani = None
        self.COM_pos_ani = None
        self.left_foot_pos_ani = None
        self.right_foot_pos_ani = None
        self.ICP_traj_ani = None
        self.ICP_pos_ani = None
        self.m_step_line_ani = None
        self.m_step_curr_ani = None

        # Delegate buffer init and _setup_figure() call to parent
        super().__init__(env, plot_cfg, plot_dir)

    # ----------------------------------------------------------
    # Abstract hook implementations
    # ----------------------------------------------------------
    def _setup_figure(self):
        """Create the three-panel figure and all artist objects."""
        self.fig = plt.figure(figsize=self.figsize)
        spec = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1.4, 1.0],
            height_ratios=[1.0, 0.55],
        )

        # 3D walking panel
        self.ax = self.fig.add_subplot(spec[:, 0], projection="3d")
        self.ax.set_xlim(*self.ax3d_xlim)
        self.ax.set_ylim(*self.ax3d_ylim)
        self.ax.set_zlim(*self.ax3d_zlim)
        self.ax.set_box_aspect(self.ax3d_box_aspect)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_zlabel("z (m)")
        self.ax.set_title("3D Walking Animation")
        self.ax.view_init(*self.ax3d_view_init)
        self.LIPM_3D_ani = LIPM_3D_Animate(self.ax)

        # Bird-eye-view panel
        self.bx = self.fig.add_subplot(spec[0, 1], autoscale_on=False)
        self.bx.set_xlim(*self.bx_xlim)
        self.bx.set_ylim(*self.bx_ylim)
        self.bx.set_aspect("equal")
        self.bx.set_xlabel("x (m)")
        self.bx.set_ylabel("y (m)")
        self.bx.set_title("Bird-eye View")
        self.bx.grid(ls="--")

        # m_step time-series panel
        self.mx = self.fig.add_subplot(spec[1, 1])
        self.mx.set_title(r"$Risk value$")
        self.mx.set_xlabel("time (s)")
        self.mx.set_ylabel(r"$Risk value")
        self.mx.grid(True, ls="--")
        self.mx.axhline(0.0, color="r", linestyle="--", linewidth=1.2)

        # Artists
        self.capture_region_ani = CircleRegion2D(
            ax=self.bx,
            radius=0.1,
            center=(0.0, 0.0),
            edgecolor="c",
            facecolor="c",
            alpha=0.15,
            linewidth=2.0,
        )
        self.ani_text_COM_pos = self.bx.text(0.05, 0.88, "", transform=self.bx.transAxes, va="top")
        self.original_ani, = self.bx.plot([0], [0], marker="o", markersize=2, color="k")
        self.COM_traj_ani, = self.bx.plot([], [], color="g")
        self.COM_pos_ani, = self.bx.plot([], [], marker="o", markersize=6, color="r")
        self.left_foot_pos_ani, = self.bx.plot([], [], "o", markersize=10, color="b")
        self.right_foot_pos_ani, = self.bx.plot([], [], "o", markersize=10, color="m")
        self.ICP_traj_ani, = self.bx.plot([], [], "--", color="k", linewidth=1.5, label="ICP traj")
        self.ICP_pos_ani, = self.bx.plot([], [], marker="x", markersize=8, color="k", label="ICP")
        self.m_step_line_ani, = self.mx.plot([], [], color="b", linewidth=2.0, label=r"$Risk_value$")
        self.m_step_curr_ani, = self.mx.plot([], [], marker="o", markersize=5, color="k")
        self.mx.legend(loc="upper right")

    def _init_func(self) -> list:
        """Initialise all artists to their frame-0 state."""
        artists_3d = self._ani_3D_init()
        artists_2d = self._ani_2D_init()
        return artists_3d + artists_2d

    def _update_func(self, i: int) -> list:
        """Update all artists to frame i."""
        artists_3d = self._ani_3D_update(i)
        artists_2d = self._ani_2D_update(i)
        return artists_3d + artists_2d

    # ----------------------------------------------------------
    # 3D animation helpers
    # ----------------------------------------------------------
    def _ani_3D_init(self) -> list:
        return []

    def _ani_3D_update(self, i: int) -> list:
        ep_start, _ = self._get_episode_range_for_frame(i)

        COM_pos = np.asarray(self._val("com_pos", i), dtype=float).reshape(3,)
        left_foot_pos = np.asarray(self._val("left_foot_pos", i), dtype=float).reshape(3,)
        right_foot_pos = np.asarray(self._val("right_foot_pos", i), dtype=float).reshape(3,)

        COM_hist = np.stack(self.buffer["com_pos"][ep_start:i + 1], axis=0)
        COM_pos_trajectory = COM_hist.T

        return self.LIPM_3D_ani.update(
            COM_pos,
            COM_pos_trajectory,
            left_foot_pos,
            right_foot_pos,
        )

    # ----------------------------------------------------------
    # 2D animation helpers
    # ----------------------------------------------------------
    def _ani_2D_init(self) -> list:
        com_pos0 = np.asarray(self._val("com_pos", 0), dtype=float).reshape(-1)
        left_foot0 = np.asarray(self._val("left_foot_pos", 0), dtype=float).reshape(-1)
        right_foot0 = np.asarray(self._val("right_foot_pos", 0), dtype=float).reshape(-1)
        icp_pos0 = np.asarray(self._val("icp_pos", 0), dtype=float).reshape(-1)

        center0 = np.asarray(self._val("capture_region_center", 0), dtype=float).reshape(2,)
        radius0 = float(self._val("capture_region_radius", 0))
        time0 = float(self._val("time_hist", 0))
        risk0 = float(self._val("risk_value", 0))
        dist0 = float(self._val("icp_ankle_dist_hist", 0))

        self.COM_traj_ani.set_data([], [])
        self.COM_pos_ani.set_data([com_pos0[0]], [com_pos0[1]])
        self.left_foot_pos_ani.set_data([left_foot0[0]], [left_foot0[1]])
        self.right_foot_pos_ani.set_data([right_foot0[0]], [right_foot0[1]])
        self.ICP_traj_ani.set_data([], [])
        self.ICP_pos_ani.set_data([icp_pos0[0]], [icp_pos0[1]])
        self.capture_region_ani.update(center=center0, radius=radius0)
        self.m_step_line_ani.set_data([], [])
        self.m_step_curr_ani.set_data([time0], [risk0])
        self.ani_text_COM_pos.set_text(
            self.COM_pos_str % (com_pos0[0], com_pos0[1], dist0, radius0, risk0)
        )

        return [
            self.COM_pos_ani,
            self.COM_traj_ani,
            self.left_foot_pos_ani,
            self.right_foot_pos_ani,
            self.ICP_pos_ani,
            self.ICP_traj_ani,
            self.capture_region_ani.artist(),
            self.ani_text_COM_pos,
            self.m_step_line_ani,
            self.m_step_curr_ani,
        ]

    def _ani_2D_update(self, i: int) -> list:
        ep_start, _ = self._get_episode_range_for_frame(i)

        com_hist = np.stack(self.buffer["com_pos"][ep_start:i + 1], axis=0)
        icp_hist = np.stack(self.buffer["icp_pos"][ep_start:i + 1], axis=0)

        com_pos = np.asarray(self._val("com_pos", i), dtype=float).reshape(-1)
        left_foot_pos = np.asarray(self._val("left_foot_pos", i), dtype=float).reshape(-1)
        right_foot_pos = np.asarray(self._val("right_foot_pos", i), dtype=float).reshape(-1)
        icp_pos = np.asarray(self._val("icp_pos", i), dtype=float).reshape(-1)

        center = np.asarray(self._val("capture_region_center", i), dtype=float).reshape(2,)
        radius = float(self._val("capture_region_radius", i))
        t_now = float(self._val("time_hist", i))
        risk_now = float(self._val("risk_value", i))
        dist_now = float(self._val("icp_ankle_dist_hist", i))

        self.COM_traj_ani.set_data(com_hist[:, 0], com_hist[:, 1])
        self.COM_pos_ani.set_data([com_pos[0]], [com_pos[1]])
        self.left_foot_pos_ani.set_data([left_foot_pos[0]], [left_foot_pos[1]])
        self.right_foot_pos_ani.set_data([right_foot_pos[0]], [right_foot_pos[1]])
        self.ICP_traj_ani.set_data(icp_hist[:, 0], icp_hist[:, 1])
        self.ICP_pos_ani.set_data([icp_pos[0]], [icp_pos[1]])
        self.capture_region_ani.update(center=center, radius=radius)

        self.ani_text_COM_pos.set_text(
            self.COM_pos_str % (com_pos[0], com_pos[1], dist_now, radius, risk_now)
        )

        if self.bx_follow_com:
            cx, cy = com_pos[0], com_pos[1]
            self.bx.set_xlim(self.bx_follow_xspan[0] + cx, self.bx_follow_xspan[1] + cx)
            self.bx.set_ylim(self.bx_follow_yspan[0] + cy, self.bx_follow_yspan[1] + cy)

        time_hist = np.asarray(self.buffer["time_hist"][ep_start:i + 1], dtype=float)
        risk_value = np.asarray(self.buffer["risk_value"][ep_start:i + 1], dtype=float)

        self.m_step_line_ani.set_data(time_hist, risk_value)
        self.m_step_curr_ani.set_data([t_now], [risk_now])

        self.mx.set_xlim(
            max(0.0, t_now - self.mx_follow_window),
            max(self.mx_initial_xlim[1], t_now + self.mx_future_margin),
        )

        y_min = float(np.min(risk_value))
        y_max = float(np.max(risk_value))
        pad = max(0.05, 0.1 * (y_max - y_min + 1e-6))
        self.mx.set_ylim(y_min - pad, y_max + pad)

        return [
            self.COM_pos_ani,
            self.COM_traj_ani,
            self.left_foot_pos_ani,
            self.right_foot_pos_ani,
            self.ICP_pos_ani,
            self.ICP_traj_ani,
            self.ani_text_COM_pos,
            self.capture_region_ani.artist(),
            self.m_step_line_ani,
            self.m_step_curr_ani,
        ]

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def show_last_frame(self):
        """Render and display the final buffered frame without saving."""
        if self.num_frames() == 0:
            raise RuntimeError("No buffered data.")
        self._finalize_open_episode()
        self._update_func(self.num_frames() - 1)
        plt.show()

    def save(self):
        """Save the animation as 'capturability.gif' inside plot_dir."""
        super().save(filename="capturability.gif")


# ============================================================
# Generic static-PNG plotter
# ============================================================
class PNGSavePlotter(GIFSavePlotter):
    """
    Static PNG plotter. Accumulates per-step trajectory data and saves
    all signals as a grid of time-series subplots in a single PNG file.

    Each key in plot_cfg becomes one subplot. Data with shape (N,) is
    averaged across the N dimension to produce a scalar time series.
    Episode boundaries are marked with vertical dashed lines.

    Compatible with the same play.py interface as GIFSavePlotter:
      plotter = PNGSavePlotter(env, plot_cfg, plot_dir)
      plotter.append(viz_data, episode_end=done)
      plotter.save()
      plotter.close()
    """

    def __init__(self, env, plot_cfg: dict, plot_dir=None):
        super().__init__(env, plot_cfg, plot_dir)

    # ----------------------------------------------------------
    # No-op animation hooks (PNG does not use FuncAnimation)
    # ----------------------------------------------------------
    def _setup_figure(self):
        """Defer figure creation to save(); nothing to build at init time."""
        self.fig = None

    def _init_func(self) -> list:
        return []

    def _update_func(self, i: int) -> list:
        return []

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _to_scalar_series(self, key: str) -> np.ndarray:
        """Convert the buffered list for key into a 1-D float array.

        Each element in the buffer may be a Python scalar or a numpy
        array of any shape.  Arrays are reduced to a single float via
        mean() so that multi-env data (shape (N,)) collapses cleanly.
        """
        return np.array(
            [float(np.asarray(v).mean()) for v in self.buffer[key]],
            dtype=float,
        )

    def _build_time_axis(self, n_frames: int) -> np.ndarray:
        """Return an x-axis array (seconds) of length n_frames.

        Uses 'time_hist' or 'time' buffer if present; otherwise falls
        back to step_index * dt.
        """
        for key in ("time_hist", "time"):
            if key in self.buffer and len(self.buffer[key]) == n_frames:
                return self._to_scalar_series(key)
        return np.arange(n_frames, dtype=float) * self.dt

    # ----------------------------------------------------------
    # PNG export
    # ----------------------------------------------------------
    def save(self, filename: str = "trajectory.png",
             excel_filename: str | None = "trajectory.xlsx"):
        """Render and save all buffered signals as a static multi-panel PNG.

        Parameters
        ----------
        filename : str
            Output PNG filename written inside self.plot_dir.
        excel_filename : str | None
            If non-None, also dumps the same buffered signals to this
            xlsx file (in self.plot_dir) via save_excel(). Pass None to
            skip the Excel export.
        """
        self._finalize_open_episode()
        n_frames = self.num_frames()
        if n_frames == 0:
            raise RuntimeError("No buffered data. Call append() first.")

        keys = list(self.cfg.keys())
        n = len(keys)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False,
        )
        self.fig = fig

        t = self._build_time_axis(n_frames)

        # Episode boundary x-positions; skip the last range to avoid
        # drawing a line at the very end of the x-axis.
        ep_boundaries = [
            t[end]
            for (_, end) in self.episode_ranges[:-1]
            if end < n_frames
        ]

        for idx, key in enumerate(keys):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            y = self._to_scalar_series(key)
            ax.plot(t, y, linewidth=1.2)
            ax.set_title(key, fontsize=8)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.grid(True, linestyle="--", linewidth=0.5)
            for xb in ep_boundaries:
                ax.axvline(xb, color="r", linestyle="--", linewidth=0.8, alpha=0.6)

        # Hide unused subplot slots in the last row
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"--------- Saved PNG to: {filepath}")

        if excel_filename is not None:
            self.save_excel(excel_filename)

    def save_excel(self, filename: str = "trajectory.xlsx"):
        """Dump buffered signals and episode boundaries to a multi-sheet xlsx.

        Two sheets are written:
          - 'trajectory': one row per step. Columns include step_index,
            time_s, episode_index, episode_step, plus every viz_data key
            expanded by component (scalar -> '{key}', length-N -> '{key}_j').
          - 'episodes': one row per episode with start/end step indices and
            timing.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "save_excel() requires pandas. Install via 'pip install pandas openpyxl'."
            ) from e
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "save_excel() requires openpyxl for .xlsx output. "
                "Install via 'pip install openpyxl'."
            ) from e

        self._finalize_open_episode()
        n_frames = self.num_frames()
        if n_frames == 0:
            raise RuntimeError("No buffered data. Call append() first.")

        t = self._build_time_axis(n_frames)

        # Per-step episode metadata
        episode_index = np.full(n_frames, -1, dtype=int)
        episode_step = np.full(n_frames, -1, dtype=int)
        for ep_i, (start, end) in enumerate(self.episode_ranges):
            episode_index[start : end + 1] = ep_i
            episode_step[start : end + 1] = np.arange(end - start + 1)

        traj: dict[str, np.ndarray] = {
            "step_index": np.arange(n_frames, dtype=int),
            "time_s": t,
            "episode_index": episode_index,
            "episode_step": episode_step,
        }
        for key in self.cfg.keys():
            arr, cols = self._to_2d_series(key)
            for j, col in enumerate(cols):
                traj[col] = arr[:, j]
        df_traj = pd.DataFrame(traj)

        ep_rows = []
        for ep_i, (start, end) in enumerate(self.episode_ranges):
            ep_rows.append({
                "episode_index": ep_i,
                "start_step": start,
                "end_step": end,
                "n_steps": end - start + 1,
                "t_start_s": float(t[start]),
                "t_end_s": float(t[end]),
                "duration_s": float(t[end] - t[start]),
            })
        df_eps = pd.DataFrame(
            ep_rows,
            columns=["episode_index", "start_step", "end_step", "n_steps",
                     "t_start_s", "t_end_s", "duration_s"],
        )

        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, filename)
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_traj.to_excel(writer, sheet_name="trajectory", index=False)
            df_eps.to_excel(writer, sheet_name="episodes", index=False)
        print(f"--------- Saved Excel to: {filepath}")

    def close(self):
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None