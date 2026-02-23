import matplotlib.pyplot as plt
import numpy as np
import torch

class LivePlotter:
    def __init__(self, env, plot_config: dict):
        self.env = env
        self.cfg = plot_config
        self.num_envs = self.env.num_envs
        self.max_length = int(self.env._unwrapped.cfg.max_episode_length)

        # Single Env
        if self.num_envs > 1:
            print(f"[LivePlotter] Warning: {self.num_envs} envs detected. Only Env 0 will be plotted for performance.")
            self.plot_idx = 0
        else:
            self.plot_idx = 0

        # Interactive mode
        plt.ion()
        num_plots = len(plot_config)
        ncols = 2
        nrows = (num_plots + 1) // 2
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(12, 2.0 * nrows))
        
        self.axs = self.axs.flatten()
        if num_plots % 2 != 0:
            self.axs[-1].axis('off')

        self.lines = {}
        self.data_buffers = {}
        self.name_to_idx = {}
        self.count = 0

        # Initialize plots
        for i, name in enumerate(plot_config.keys()):
            self.name_to_idx[name] = i
            self.lines[name] = []
            self.data_buffers[name] = []

            line, = self.axs[i].plot(np.full(self.max_length, np.nan), color='royalblue', linewidth=1.5)
            self.lines[name].append(line)
            self.data_buffers[name].append(np.full(self.max_length, np.nan))
            
            self.axs[i].set_title(name, fontsize=10, pad=2)
            self.axs[i].grid(True, alpha=0.2)
            self.axs[i].set_xlim(0, self.max_length)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)


    def reset(self):
        """
        Re-intialization at every Env reset
        NOTE: This reset function assumes single environment
        """
        for name in self.data_buffers:
            for i in range(len(self.data_buffers[name])):
                self.data_buffers[name][i].fill(np.nan)
                self.lines[name][i].set_ydata(self.data_buffers[name][i])
        
        self.count = 0

        for ax in self.axs:
            ax.relim()     
            ax.autoscale_view()
        
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.flush_events()
    

    def update(self, viz_data: dict):
        """
        Update Plots at every Env step
        NOTE: This update function assumes single environment
        """
        self.count += 1  
        needs_full_draw = False
        
        for name, values in viz_data.items():
            idx = self.name_to_idx[name]
            ax = self.axs[idx]
            
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
                
                self.lines[name][i].set_ydata(self.data_buffers[name][i])

            # Y-Range
            all_data = np.concatenate(self.data_buffers[name])
            valid_data = all_data[~np.isnan(all_data)] 
            if len(valid_data) > 0:
                y_min, y_max = np.min(valid_data), np.max(valid_data)
                curr_min, curr_max = ax.get_ylim()
                if y_min < curr_min or y_max > curr_max:
                    ax.set_ylim(y_min - abs(y_min)*0.1 - 0.1, y_max + abs(y_max)*0.1 + 0.1)
                    needs_full_draw = True

        # --- 나머지 Blitting 로직은 동일 ---
        if needs_full_draw:
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            for name, lines in self.lines.items():
                ax_idx = self.name_to_idx[name]
                for line in lines:
                    self.axs[ax_idx].draw_artist(line)
            self.fig.canvas.blit(self.fig.bbox)
        
        self.fig.canvas.flush_events()