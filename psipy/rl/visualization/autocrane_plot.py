import numpy as np
from matplotlib import pyplot as plt

from psipy.core.notebook_tools import is_notebook
from psipy.rl.io.batch import Episode

class AutocraneTrolleyTrajectoryPlot:
    def __init__(self, 
                 filename=None,
                 trolley_position_idx: int = 0,
                 trolley_vel_idx: int = 1,
                 trolley_sway_idx: int = 2,
                 trolley_sway_vel_idx: int = 3,
                 hoist_position_idx: int = 4,
                 hoist_vel_idx: int = 5,
                 do_plot_hoist: bool = False,
                 do_display: bool = True,
                 trolley_margin = 0.3,
                 hoist_margin = 0.1,
                 sway_margin = 0.04):
        self.trolley_position_idx = trolley_position_idx
        self.trolley_vel_idx = trolley_vel_idx
        self.trolley_sway_idx = trolley_sway_idx
        self.trolley_sway_vel_idx = trolley_sway_vel_idx
        self.hoist_position_idx = hoist_position_idx
        self.hoist_vel_idx = hoist_vel_idx

        self.trolley_margin = trolley_margin
        self.hoist_margin = hoist_margin
        self.sway_margin = sway_margin

        self.do_plot_hoist = do_plot_hoist
        self.episode = None
        self.filename = filename
        self.fig = None
        self.axs = None
        self.dirty = True
        self.do_display = do_display
        self.episode_num = None
        self.title_string = None

    def update(self, episode: Episode,
               episode_num: int = None, title_string: str = None):
        self.episode = episode
        self.episode_num = episode_num
        self.title_string = title_string
        self.dirty = True

    def plot(self):
        self._maybe_plot()
    
    def save(self, filename=None):
        filename = self.filename if filename is None else filename
        self._maybe_plot()
        self.fig.savefig(filename)

    def _is_notebook(self):
        return is_notebook()  # there is a shared implementation provided by psipy.core.notebook_tools

    def _maybe_plot(self):
        if not self.dirty:
            return

        if self.fig is None:
            if self.do_plot_hoist:
                self.fig, self.axs = plt.subplots(6, figsize=(10, 10))
            else:
                self.fig, self.axs = plt.subplots(4, figsize=(10, 8))
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        elif not self._is_notebook():
            plt.figure(self.fig.number)
            
        for ax in self.axs:
            ax.clear()

        x = self.episode.observations[:, self.trolley_position_idx]
        x_s = self.episode.observations[:, self.trolley_vel_idx]
        # t = self.episode.observations[:, self.pole_theta_idx]
        sway = self.episode.observations[:, self.trolley_sway_idx]
        a = self.episode._actions[:, 0]
        cost = self.episode.costs

        if self.do_plot_hoist:
            hoist_pos = self.episode.observations[:, self.hoist_position_idx]
            hoist_vel = self.episode.observations[:, self.hoist_vel_idx]
            hoist_action = self.episode._actions[:, 1]

        axis_counter = 0


        self.axs[axis_counter].plot(x, label="trolley_position")
        self.axs[axis_counter].axhline(0, color="grey", linestyle=":", label="target")
        # Add blue shaded area of ±0.4m around target to show acceptable deviation
        self.axs[axis_counter].axhspan(-self.trolley_margin, 
                                       self.trolley_margin, 
                                       color='grey', alpha=0.2, label='acceptable range')
        # Add darker blue lines at boundaries
        self.axs[axis_counter].axhline(-self.trolley_margin, color='grey', alpha=0.5)
        self.axs[axis_counter].axhline(self.trolley_margin, color='grey', alpha=0.5)
        self.axs[axis_counter].set_title("trolley_position")
        self.axs[axis_counter].set_ylabel("Position")
        self.axs[axis_counter].set_ylim((-4.5, 4.5))
        self.axs[axis_counter].legend()
        axis_counter += 1
        
        if self.do_plot_hoist:
            self.axs[axis_counter].plot(hoist_pos, label="hoist_position")
            self.axs[axis_counter].axhline(0, color="grey", linestyle=":", label="target")
            # Add blue shaded area of ±0.4m around target to show acceptable deviation
            self.axs[axis_counter].axhspan(-self.hoist_margin, 
                                           self.hoist_margin, 
                                           color='grey', alpha=0.2, label='acceptable range')
            # Add darker blue lines at boundaries
            self.axs[axis_counter].axhline(-self.hoist_margin, color='grey', alpha=0.5)
            self.axs[axis_counter].axhline(self.hoist_margin, color='grey', alpha=0.5)
            self.axs[axis_counter].set_ylim((-1.0, 1.0))
            self.axs[axis_counter].set_title("hoist_position")
            self.axs[axis_counter].set_ylabel("Position")
            self.axs[axis_counter].legend()
            axis_counter += 1

        self.axs[axis_counter].plot(sway, label="sway")
        self.axs[axis_counter].axhline(-self.sway_margin, color="grey", linestyle=":", label="target")
        self.axs[axis_counter].axhline(self.sway_margin, color="grey", linestyle=":")
        self.axs[axis_counter].set_title("sway")
        self.axs[axis_counter].set_ylabel("Sway")
        self.axs[axis_counter].set_ylim((-0.15, 0.15))
        self.axs[axis_counter].legend()
        axis_counter += 1

        self.axs[axis_counter].plot(a, label="trolley action")
        self.axs[axis_counter].plot(x_s, color="black", alpha=0.4, label="trolley_velocity")
        self.axs[axis_counter].axhline(0, color="grey", linestyle=":")
        self.axs[axis_counter].set_title("Control")
        self.axs[axis_counter].set_ylabel("Velocity")
        self.axs[axis_counter].set_ylim((-0.5, 0.5))
        self.axs[axis_counter].legend(loc="upper left")
        axis_counter += 1

        if self.do_plot_hoist:
            self.axs[axis_counter].plot(hoist_action, label="hoist action")
            self.axs[axis_counter].plot(hoist_vel, color="black", alpha=0.4, label="hoist_velocity")
            self.axs[axis_counter].axhline(0, color="grey", linestyle=":")
            self.axs[axis_counter].set_title("Hoist Control")
            self.axs[axis_counter].set_ylabel("Velocity")
            self.axs[axis_counter].set_ylim((-0.2, 0.2))
            self.axs[axis_counter].legend(loc="upper left")
            axis_counter += 1
            

        if cost is not None:
            self.axs[axis_counter].plot(cost, label="cost")
            self.axs[axis_counter].set_title("cost")
            self.axs[axis_counter].set_ylabel("cost")
            self.axs[axis_counter].set_ylim((0.0, 0.03))
            self.axs[axis_counter].legend()
            axis_counter += 1

        if self.episode_num is None:
            title = "Trolley Control"
        else:
            title = "Trolley Control, Episode {}".format(self.episode_num)

        if self.title_string:
            title = title + " - " + self.title_string

        self.fig.suptitle(title)
   
    
        if self._is_notebook():
            self.fig.canvas.draw()
        else:
            # This is what makes it live
            # If you get
            #   AttributeError: type object 'FigureCanvasBase'
            #     has no attribute 'start_event_loop_default'
            # you are in a notebook and either you did not set
            # 'in_notebook' to True, or it wasn't detected correctly.
            plt.pause(0.01)      

        self.dirty = False

        

