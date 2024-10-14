import numpy as np
from matplotlib import pyplot as plt

from psipy.core.notebook_tools import is_notebook
from psipy.rl.io.batch import Episode

class CartPoleTrajectoryPlot:
    def __init__(self, 
                 filename=None,
                 cart_position_idx: int = 0,
                 cart_velocity_idx: int = 1,
                 pole_sine_idx: int = 3,
                 pole_cosine_idx: int = 4,
                 pole_velocity_idx: int = 5,
                 do_display: bool = True):
        self.cart_position_idx = cart_position_idx
        self.cart_velocity_idx = cart_velocity_idx
        self.pole_sine_idx = pole_sine_idx
        self.pole_cosine_idx = pole_cosine_idx
        self.pole_velocity_idx = pole_velocity_idx

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
            self.fig, self.axs = plt.subplots(5, figsize=(10, 8))
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        elif not self._is_notebook():
            plt.figure(self.fig.number)
            
        for ax in self.axs:
            ax.clear()

        x = self.episode.observations[:, self.cart_position_idx]
        x_s = self.episode.observations[:, self.cart_velocity_idx]
        # t = self.episode.observations[:, self.pole_theta_idx]
        pole_sine = self.episode.observations[:, self.pole_sine_idx]
        pole_cosine = self.episode.observations[:, self.pole_cosine_idx]
        td = self.episode.observations[:, self.pole_velocity_idx]
        a = self.episode._actions[:, 0]
        cost = self.episode.costs
    

        self.axs[0].plot(x, label="cart_position")
        self.axs[0].set_title("cart_position")
        self.axs[0].set_ylabel("Position")
        self.axs[0].legend()

        self.axs[1].plot(pole_cosine, label="cos")
        self.axs[1].plot(pole_sine, label="sin")
        self.axs[1].axhline(0, color="grey", linestyle=":", label="target")
        self.axs[1].set_title("Angle")
        #self.axs[1].set_ylim((-1.0, 1,0))
        self.axs[1].set_ylabel("Angle")
        self.axs[1].legend()

        self.axs[2].plot(td, label="pole_velocity")
        self.axs[2].set_title("pole_velocity")
        self.axs[2].set_ylabel("Angular Vel")
        self.axs[2].legend()

        self.axs[3].plot(a, label="Action")
        self.axs[3].axhline(0, color="grey", linestyle=":")
        self.axs[3].set_title("Control")
        self.axs[3].set_ylabel("Velocity")
        self.axs[3].legend(loc="upper left")
     #   axes2b = axs[3].twinx()
     #   axes2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
     #   axes2b.set_ylabel("Steps/s")
     #   axes2b.legend(loc="upper right")

        if cost is not None:
            self.axs[4].plot(cost, label="cost")
            self.axs[4].set_title("cost")
            self.axs[4].set_ylabel("cost")
            self.axs[4].legend()

        if self.episode_num is None:
            title = "Cartpole"
        else:
            title = "Cartpole, Episode {}".format(self.episode_num)

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

        

