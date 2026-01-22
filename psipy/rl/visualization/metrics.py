import numpy as np
from matplotlib import pyplot as plt

from psipy.core.notebook_tools import is_notebook


class RLMetricsPlot:
    def __init__(self, metrics=None, filename=None):
        self.metrics = metrics if metrics is not None else { 
            "total_cost": [], 
            "avg_cost": [], 
            "cycles_run": [],
            "wall_time_s": [] 
        }
        self.filename = filename
        self.fig = None
        self.ax = None
        self.dirty = True
        self.window_size = 7

    def update(self, metrics):
        self.metrics = metrics
        self.dirty = True

    def plot(self):
        self._maybe_plot()
    
    def save(self, filename=None):
        filename = self.filename if filename is None else filename
        self._maybe_plot()
        if self.fig is None or filename is None:
            print("No figure or no filename to save metrics plot")
            return
        self.fig.savefig(filename)

    def _is_notebook(self):
        return is_notebook()  # there is a shared implementation provided by psipy.core.notebook_tools

    def _maybe_plot(self):
        if not self.dirty:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        elif not self._is_notebook():
            plt.figure(self.fig.number)

        if self.window_size > len(self.metrics["avg_cost"]):
            return
            
        self.ax.clear()
    
        #print(">>> metrics['avg_cost']", metrics["avg_cost"])
    
        # Calculate moving average and variance
        avg_cost = np.array(self.metrics["avg_cost"])
        moving_avg = np.convolve(avg_cost, np.ones(self.window_size)/self.window_size, mode='same')
    
        # Calculate moving variance
        moving_var = np.convolve(avg_cost**2, np.ones(self.window_size)/self.window_size, mode='same') - moving_avg**2
        moving_std = np.sqrt(moving_var)
    
        # Plot original data, moving average, and variance
        x = range(len(avg_cost))
        x_valid = x # range(window_size-1, len(avg_cost))
    
        self.ax.plot(x_valid, avg_cost, label="avg_cost", alpha=0.3, color='gray')
        self.ax.plot(x_valid, moving_avg, label="moving average", color='blue')
        self.ax.fill_between(x_valid, moving_avg - moving_std, moving_avg + moving_std, alpha=0.2, color='blue', label='±1 std dev')
    
        self.ax.set_title("Average Cost")
        self.ax.set_ylabel("Cost per step")
        self.ax.legend()
        

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

