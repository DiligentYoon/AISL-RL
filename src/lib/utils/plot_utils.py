from __future__ import annotations

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import torch
import sys

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from abc import abstractmethod

class PyQtLivePlotter:
    def __init__(self, env, plot_config: dict):
        self.env = env
        self.cfg = plot_config
        self.num_envs = self.env.num_envs
        self.max_length = int(self.env._unwrapped.cfg.max_episode_length)

        # Single Env Safety
        if self.num_envs > 1:
            print(f"[LivePlotter] Warning: {self.num_envs} envs detected. Only Env 0 will be plotted for performance.")
            self.plot_idx = 0
        else:
            self.plot_idx = 0

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        pg.setConfigOptions(antialias=True) 

        self.win = pg.GraphicsLayoutWidget(show=True, title="Isaac Lab Live Monitor")
        num_plots = len(plot_config)
        nrows = (num_plots + 1) // 2
        self.win.resize(1200, 300 * nrows) 

        self.plots = {}
        self.curves = {}
        self.data_buffers = {}
        self.name_to_idx = {}
        self.count = 0

        ncols = 2
        for i, name in enumerate(plot_config.keys()):
            self.name_to_idx[name] = i
            self.curves[name] = []
            self.data_buffers[name] = []

            row = i // ncols
            col = i % ncols
            p = self.win.addPlot(row=row, col=col, title=name)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setXRange(0, self.max_length)
            
            p.enableAutoRange(axis='y', enable=True)
            self.plots[name] = p

            curve = p.plot(pen=pg.mkPen('royalblue', width=2))
            self.curves[name].append(curve)
            self.data_buffers[name].append(np.full(self.max_length, np.nan))

    def reset(self):
        """
        Re-initialization at every Env reset
        """
        for name in self.data_buffers:
            for i in range(len(self.data_buffers[name])):
                self.data_buffers[name][i].fill(np.nan)
                self.curves[name][i].setData(self.data_buffers[name][i])
        
        self.count = 0

        for p in self.plots.values():
            p.enableAutoRange(axis='y', enable=True)
            
        self.app.processEvents()

    def update(self, viz_data: dict):
        """
        Update Plots at every Env step
        """
        self.count += 1  
        
        for name, values in viz_data.items():
            if name not in self.curves:
                continue
            
            for i, val in enumerate(values):
                if torch.is_tensor(val) and val.dim() > 0:
                    new_val = val[self.plot_idx].item()
                else:
                    new_val = val.item() if torch.is_tensor(val) else val

                if self.count < self.max_length:
                    self.data_buffers[name][i][self.count - 1] = new_val
                else:
                    self.data_buffers[name][i] = np.roll(self.data_buffers[name][i], -1)
                    self.data_buffers[name][i][-1] = new_val
                
                self.curves[name][i].setData(self.data_buffers[name][i], connect='finite')

        self.app.processEvents()

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
# Main plotter
# ============================================================
class GIFSavePlotter:
    def __init__(self, env, plot_cfg: dict, plot_dir):
        self.env = env
        self.plot_cfg = plot_cfg
        self.dt = self.env._unwrapped.step_dt    
        self.plot_dir = plot_dir if plot_dir is not None else os.getcwd()    
        
    @abstractmethod
    def reset(self):
        raise NotImplementedError(f"Please implement the 'reset' method for {self.__class__.__name__}.")
        

    @abstractmethod
    def append(self):
        raise NotImplementedError(f"Please implement the 'append' method for {self.__class__.__name__}.")

    @abstractmethod
    def save(self):
        raise NotImplementedError(f"Please implement the 'save' method for {self.__class__.__name__}.")


class CapturabilityPlotter(GIFSavePlotter):
    """
    Streaming plotter for single-env capturability visualization.

    Notes
    -----
    1. Incoming viz_data is one-step data for the already-selected environment.
    2. Keys of plot_cfg["viz_data"] define the internal buffer schema.
    3. Multiple episodes can be concatenated into a single GIF.
    4. Episode boundary is given externally through episode_end=True.
    """

    REQUIRED_KEYS = (
        "com_pos",
        "left_foot_pos",
        "right_foot_pos",
        "icp_pos",
        "capture_region_center",
        "capture_region_radius",
        "time_hist",
        "m_step_hist",
        "icp_ankle_dist_hist",
    )

    def __init__(self, env, plot_cfg: dict, plot_dir = None):
        super().__init__(env, plot_cfg, plot_dir)

        missing = [k for k in self.REQUIRED_KEYS if k not in plot_cfg]
        if missing:
            raise KeyError(f"plot_cfg['viz_data'] missing required keys: {missing}")

        self.dt = self.env._unwrapped.step_dt

        # Figure size
        self.figsize =(14, 10)
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
        self.mx_initial_xlim  = (0.0, 3.0)
        self.mx_follow_window = 2.0
        self.mx_future_margin = 0.2

        # text
        self.COM_pos_str = "COM = (%.2f, %.2f)\nICP-ankle = %.3f m\nradius = %.3f m\nm_step = %.3f m"
        
        # buffer schema = directly from viz_data keys
        self.buffer = {key: [] for key in plot_cfg.keys()}

        # matplotlib handles
        self.fig = None
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

        # multi-episode support
        self.episode_ranges: list[tuple[int, int]] = []
        self.current_episode_start = 0

        self._setup_figure()

    # ------------------------------------------------------------
    # buffer / input
    # ------------------------------------------------------------
    def reset(self):
        for key in self.buffer:
            self.buffer[key].clear()
        self.episode_ranges.clear()
        self.current_episode_start = 0

    def num_frames(self) -> int:
        first_key = next(iter(self.buffer.keys()))
        return len(self.buffer[first_key])

    def append(self, viz_data: dict, episode_end: bool = False):
        """
        Append one step of data.

        Parameters
        ----------
        viz_data : dict
            One-step visualization data for the selected environment.
        episode_end : bool
            True if this appended frame is the last frame of the current episode.
        """
        for key in self.buffer.keys():
            if key not in viz_data:
                raise KeyError(f"Missing key in viz_data: '{key}'")
            self.buffer[key].append(self.process_data(viz_data[key]))

        curr_idx = self.num_frames() - 1

        if episode_end:
            self.episode_ranges.append((self.current_episode_start, curr_idx))
            self.current_episode_start = curr_idx + 1

    def process_data(self, x):
        """
        Convert incoming data to numpy/scalar form.

        - torch.Tensor -> numpy
        - scalar-like  -> python scalar
        - array-like   -> copied ndarray
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x)

        if x.ndim == 0:
            return x.item()
        return x.copy()

    # ------------------------------------------------------------
    # figure setup
    # ------------------------------------------------------------
    def _setup_figure(self):
        self.fig = plt.figure(figsize=self.figsize)
        spec = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1.4, 1.0],
            height_ratios=[1.0, 0.55],
        )

        # 3D animation panel
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

        # M function metric panel
        self.mx = self.fig.add_subplot(spec[1, 1])
        self.mx.set_title(r"$m_{step}(t)=d^{(2)}_{avail}(t)-\|r_{ic}(t)-r_{ankle}\|$")
        self.mx.set_xlabel("time (s)")
        self.mx.set_ylabel(r"$m_{step}$ (m)")
        self.mx.grid(True, ls="--")
        self.mx.axhline(0.0, color="r", linestyle="--", linewidth=1.2)

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

        self.m_step_line_ani, = self.mx.plot([], [], color="b", linewidth=2.0, label=r"$m_{step}(t)$")
        self.m_step_curr_ani, = self.mx.plot([], [], marker="o", markersize=5, color="k")
        self.mx.legend(loc="upper right")

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _val(self, key, i):
        return self.buffer[key][i]

    def _get_episode_range_for_frame(self, i: int):
        for start, end in self.episode_ranges:
            if start <= i <= end:
                return start, end
        raise IndexError(f"Frame index {i} is not covered by any episode range.")

    def _finalize_open_episode(self):
        n = self.num_frames()
        if n == 0:
            return

        last_closed_end = -1 if len(self.episode_ranges) == 0 else self.episode_ranges[-1][1]

        if last_closed_end < n - 1:
            self.episode_ranges.append((self.current_episode_start, n - 1))
            self.current_episode_start = n

    # ------------------------------------------------------------
    # 3D animation functions
    # ------------------------------------------------------------
    def ani_3D_init(self):
        return []

    def ani_3D_update(self, i):
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

    # ------------------------------------------------------------
    # 2D animation functions
    # ------------------------------------------------------------
    def ani_2D_init(self):
        com_pos0 = np.asarray(self._val("com_pos", 0), dtype=float).reshape(-1)
        left_foot0 = np.asarray(self._val("left_foot_pos", 0), dtype=float).reshape(-1)
        right_foot0 = np.asarray(self._val("right_foot_pos", 0), dtype=float).reshape(-1)
        icp_pos0 = np.asarray(self._val("icp_pos", 0), dtype=float).reshape(-1)

        center0 = np.asarray(self._val("capture_region_center", 0), dtype=float).reshape(2,)
        radius0 = float(self._val("capture_region_radius", 0))

        time0 = float(self._val("time_hist", 0))
        mstep0 = float(self._val("m_step_hist", 0))
        dist0 = float(self._val("icp_ankle_dist_hist", 0))

        self.COM_traj_ani.set_data([], [])
        self.COM_pos_ani.set_data([com_pos0[0]], [com_pos0[1]])
        self.left_foot_pos_ani.set_data([left_foot0[0]], [left_foot0[1]])
        self.right_foot_pos_ani.set_data([right_foot0[0]], [right_foot0[1]])

        self.ICP_traj_ani.set_data([], [])
        self.ICP_pos_ani.set_data([icp_pos0[0]], [icp_pos0[1]])

        self.capture_region_ani.update(center=center0, radius=radius0)

        self.m_step_line_ani.set_data([], [])
        self.m_step_curr_ani.set_data([time0], [mstep0])

        self.ani_text_COM_pos.set_text(
            self.COM_pos_str % (
                com_pos0[0],
                com_pos0[1],
                dist0,
                radius0,
                mstep0,
            )
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

    def ani_2D_update(self, i):
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
        m_step_now = float(self._val("m_step_hist", i))
        dist_now = float(self._val("icp_ankle_dist_hist", i))

        self.COM_traj_ani.set_data(com_hist[:, 0], com_hist[:, 1])
        self.COM_pos_ani.set_data([com_pos[0]], [com_pos[1]])
        self.left_foot_pos_ani.set_data([left_foot_pos[0]], [left_foot_pos[1]])
        self.right_foot_pos_ani.set_data([right_foot_pos[0]], [right_foot_pos[1]])

        self.ICP_traj_ani.set_data(icp_hist[:, 0], icp_hist[:, 1])
        self.ICP_pos_ani.set_data([icp_pos[0]], [icp_pos[1]])

        self.capture_region_ani.update(center=center, radius=radius)

        self.ani_text_COM_pos.set_text(
            self.COM_pos_str % (
                com_pos[0],
                com_pos[1],
                dist_now,
                radius,
                m_step_now,
            )
        )

        if self.bx_follow_com:
            cx, cy = com_pos[0], com_pos[1]
            self.bx.set_xlim(self.bx_follow_xspan[0] + cx, self.bx_follow_xspan[1] + cx)
            self.bx.set_ylim(self.bx_follow_yspan[0] + cy, self.bx_follow_yspan[1] + cy)

        time_hist = np.asarray(self.buffer["time_hist"][ep_start:i + 1], dtype=float)
        m_step_hist = np.asarray(self.buffer["m_step_hist"][ep_start:i + 1], dtype=float)

        self.m_step_line_ani.set_data(time_hist, m_step_hist)
        self.m_step_curr_ani.set_data([t_now], [m_step_now])

        self.mx.set_xlim(
            max(0.0, t_now - self.mx_follow_window),
            max(self.mx_initial_xlim[1], t_now + self.mx_future_margin),
        )

        y_min = float(np.min(m_step_hist))
        y_max = float(np.max(m_step_hist))
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

    # ------------------------------------------------------------
    # animation wrappers
    # ------------------------------------------------------------
    def _init_func(self):
        artist1 = self.ani_3D_init()
        artist2 = self.ani_2D_init()
        return artist1 + artist2

    def _update_func(self, i):
        artist1 = self.ani_3D_update(i)
        artist2 = self.ani_2D_update(i)
        return artist1 + artist2

    # ------------------------------------------------------------
    # utility
    # ------------------------------------------------------------
    def show_last_frame(self):
        if self.num_frames() == 0:
            raise RuntimeError("No buffered data.")
        self._finalize_open_episode()
        self._update_func(self.num_frames() - 1)
        plt.show()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    # ------------------------------------------------------------
    # save
    # ------------------------------------------------------------
    def save(self):
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

        fps = max(1, int((1.0 / self.dt)))
        writer = PillowWriter(fps=fps)

        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, "capturability.gif")
        anim.save(filepath, writer=writer)
        print(f"--------- Saved animation to: {filepath}")