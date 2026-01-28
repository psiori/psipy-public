import numpy as np
from matplotlib import pyplot as plt

from psipy.core.notebook_tools import is_notebook


class RLMetricsPlot:
    def __init__(self, metrics=None, filename=None):
        self.metrics = (
            metrics
            if metrics is not None
            else {
                "total_cost": [],
                "avg_cost": [],
                "cycles_run": [],
                "wall_time_s": [],
            }
        )
        self.filename = filename
        self.fig = None
        self.ax = None
        self.dirty = True
        self.window_size = 5

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
        self.fig.tight_layout()
        self.fig.savefig(filename, bbox_inches="tight", pad_inches=0.1)

    def _is_notebook(self):
        return (
            is_notebook()
        )  # there is a shared implementation provided by psipy.core.notebook_tools

    def _maybe_plot(self):
        if not self.dirty:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        elif not self._is_notebook():
            plt.figure(self.fig.number)
            if self.ax is None:
                self.ax = self.fig.axes[0] if self.fig.axes else None

        if self.window_size > len(self.metrics["avg_cost"]):
            return

        self.ax.clear()

        # Get tab10 color scheme
        cmap = plt.cm.get_cmap("tab10")
        color0 = cmap(0)  # Blue
        color1 = cmap(1)  # Orange

        # print(">>> metrics['avg_cost']", metrics["avg_cost"])

        # Calculate moving average and variance
        avg_cost = np.array(self.metrics["avg_cost"])

        # Use current value and n preceding values (or all preceding if fewer than n)
        moving_avg = np.zeros_like(avg_cost)
        moving_std = np.zeros_like(avg_cost)

        for i in range(len(avg_cost)):
            # Use current value and up to window_size preceding values
            print(self.window_size)
            start = max(0, i - self.window_size + 1)
            end = i + 1
            print(start, end)
            window_data = avg_cost[start:end]

            # Calculate mean and std for this window
            moving_avg[i] = np.mean(window_data)
            moving_std[i] = np.std(window_data)

        # Plot original data, moving average, and variance
        x = range(len(avg_cost))
        x_valid = x  # range(window_size-1, len(avg_cost))

        self.ax.plot(
            x_valid,
            avg_cost,
            label="Durchschnittskosten",
            alpha=0.3,
            color=color1,
        )
        self.ax.plot(x_valid, moving_avg, label="Gleitender Mittelwert", color=color0)
        self.ax.fill_between(
            x_valid,
            moving_avg - moving_std,
            moving_avg + moving_std,
            alpha=0.2,
            color=color0,
            label="±1 Standardabweichung",
        )

        # self.ax.set_title("Durchschnittskosten")
        self.ax.set_ylabel("Kosten pro Zeitschritt")
        self.ax.set_xlabel("Episoden")
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
