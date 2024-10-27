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
                 do_display: bool = True):
        self.trolley_position_idx = trolley_position_idx
        self.trolley_vel_idx = trolley_vel_idx
        self.trolley_sway_idx = trolley_sway_idx

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

        self.axs[0].plot(x, label="trolley_position")
        self.axs[0].set_title("trolley_position")
        self.axs[0].set_ylabel("Position")
        self.axs[0].legend()

        self.axs[1].plot(sway, label="sway")
        self.axs[1].axhline(0, color="grey", linestyle=":", label="target")
        self.axs[1].set_title("sway")
        self.axs[1].set_ylim((-0.5, 0.5))
        self.axs[1].set_ylabel("Sway")
        self.axs[1].legend()

        self.axs[2].plot(a, label="action")
        self.axs[2].plot(x_s, color="black", alpha=0.4, label="trolley_velocity")
        self.axs[2].axhline(0, color="grey", linestyle=":")
        self.axs[2].set_title("Control")
        self.axs[2].set_ylabel("Velocity")
        self.axs[2].legend(loc="upper left")

        if cost is not None:
            self.axs[3].plot(cost, label="cost")
            self.axs[3].set_title("cost")
            self.axs[3].set_ylabel("cost")
            self.axs[3].legend()

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

        

