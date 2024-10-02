# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Callbacks for Keras which involve plotting the course of training.

Keras accepts callbacks which are activated at various points in training,
such as at batch end or epoch end.  In order to visualize live training,
give the .fit() of your model the PlottingCallback.  This will plot your
training live in a separate window.

If you are using the callback in a notebook, you MUST set in_notebook to True!

Below is an example set of callbacks for training NFQCA, which has two networks,
a critic and an actor.  Thus, there are two plots::

    critic_callback = PlottingCallback(
        ax1="q",
        is_ax1=lambda x: x.endswith("q") or "act" in x,
        ax2="mse",
        is_ax2=lambda x: x == "loss",
        title="Critic",
    )

    actor_callback = PlottingCallback(
        ax1="act",
        is_ax1=lambda x: "act" in x,
        ax2="mse",
        is_ax2=lambda x: x == "loss",
        title="Actor",
    )

.. autosummary::

    COLORS
    PlottingCallback

"""

import os
from itertools import cycle
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import MaxNLocator

from psipy.core.notebook_tools import is_notebook

__all__ = ["COLORS", "PlottingCallback", "QVizCallback"]


#: Some generic but pretty matplotlib colors. See the `matplotlib docs \
#: <https://matplotlib.org/3.1.1/api/colors_api.html>`_ for more information.
COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


class PlottingCallback(tf.keras.callbacks.Callback):
    """Keras callback for live plotting during training.

    Args:
        fig: the figure to plot on; if None, creates own figure
        ax: the axis to plot on; if None, creates own axis
        ax1: y axis name for axis 1
        ax2: y axis name for axis 2
        is_ax1: function to extract variables to plot in axis 1
        is_ax2: function to extract variables to plot in axis 2
        title: Title for the plot figure.
        filepath: Path to store the plot on disk. Stored image will always be
                  overwritten with the latest state.
        save_freq: When saving, whether to save the image at the end of every
                  'epoch' or 'end'.
        plot_freq: When to plot, either at the end of every 'epoch' or at the
                   end of every iteration ('end').
        dims: Tuple of canvas dimensions; defaults to (10,4)
    """

    def __init__(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        ax1: Optional[str] = None,
        ax2: Optional[str] = None,
        is_ax1: Optional[Callable[[str], bool]] = None,
        is_ax2: Optional[Callable[[str], bool]] = None,
        title: Optional[str] = None,
        filepath: Optional[str] = None,
        save_freq: str = "epoch",
        plot_freq: str = "epoch",
        dims: Tuple[int, int] = (10, 4),
    ):
        self.filepath = filepath
        assert save_freq in ["epoch", "end"]
        assert plot_freq in ["epoch", "end"]
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        self.in_notebook = is_notebook()

        # Cycle through the predefined set of colors indefinitely.
        self._colors = cycle(COLORS)

        # Only turn interactive mode on if we have a display.
        if os.environ.get("DISPLAY", False):
            plt.ion()

        if ax is not None or fig is not None:
            assert ax is not None and fig is not None, "Must provide both!"
            self.fig = fig
            self.ax = ax
        else:
            self.fig = plt.figure(figsize=dims)
            self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.ax.set_xlabel("epoch")
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        self._has_ax2 = False
        self._is_ax2 = lambda m: False
        if ax2 is not None:
            self.ax2.set_ylabel(ax2)
            self.ax2.set_yscale('log')
            self._has_ax2 = True
            self._is_ax2 = lambda m: m == ax2
        if is_ax2 is not None:
            self._has_ax2 = True
            self._is_ax2 = is_ax2

        self._is_ax1 = lambda m: not self._is_ax2(m)
        if ax1 is not None:
            self.ax.set_ylabel(ax1)
            self.ax.set_yscale('log')
            self._is_ax1 = lambda m: m == ax1
        if is_ax1 is not None:
            self._is_ax1 = is_ax1

        if title is not None:
            self.fig.suptitle(title)

        self.legend = None
        self.metrics: Dict[str, Any] = {}
        self.plots: Dict[str, Any] = {}
        self.epochs = 0
        if self.in_notebook:
            plt.show()

    def _plot(self, metrics):
        if not self.in_notebook:
            # Activate the relevant figure
            plt.figure(self.fig.number)

        epochs = list(range(1, self.epochs + 1))
        for metric, values in self.metrics.items():
            self.plots[metric].set_data(epochs, values)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        if self.legend is not None:
            self.legend.remove()
        self.legend = self.fig.legend(
            *list(
                zip(
                    *[
                        (
                            self.plots[metric],
                            "{}: {:.6f}".format(metric, self.metrics[metric][-1]),
                        )
                        for metric in self.metrics.keys()
                    ]
                )
            ),
            loc=2,
        )
        if self.in_notebook:
            self.fig.canvas.draw()
        else:
            # This is what makes it live
            # If you get
            #   AttributeError: type object 'FigureCanvasBase'
            #     has no attribute 'start_event_loop_default'
            # you are in a notebook and either you did not set
            # 'in_notebook' to True, or it wasn't detected correctly.
            plt.pause(0.01)

    def on_epoch_end(self, epoch, logs=None):
        # Record the data coming in
        self.epochs += 1
        for metric, value in logs.items():
            if metric not in self.metrics:
                if self._is_ax2(metric):
                    self.metrics[metric] = []
                    (self.plots[metric],) = self.ax2.plot(
                        [0], [0], color=next(self._colors)
                    )
                if self._is_ax1(metric):
                    self.metrics[metric] = []
                    (self.plots[metric],) = self.ax.plot(
                        [0], [0], color=next(self._colors)
                    )
            if metric in self.metrics:
                self.metrics[metric].append(value)

        # Perform actions if desired
        if self.plot_freq == "epoch":
            self._plot(metrics=logs)
        if self.filepath is not None and self.save_freq == "epoch":
            self.fig.savefig(
                self.filepath, dpi=self.fig.dpi, bbox_inches="tight", pad_inches=0
            )

    def on_train_end(self, logs=None):
        if self.plot_freq == "end":
            self._plot(metrics=logs)
        if self.filepath is not None and self.save_freq == "end":
            self.fig.savefig(
                self.filepath, dpi=self.fig.dpi, bbox_inches="tight", pad_inches=0
            )


class QVizCallback(tf.keras.callbacks.Callback):
    def __init__(self, title: Optional[str] = None, filepath: Optional[str] = None):
        """Keras callback for live Q transition plotting during training """
        plt.ion()

        self.filepath = filepath
        self.in_notebook = is_notebook()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_xlabel("Transition")

        if title is not None:
            self.fig.suptitle(title)
        else:
            self.fig.suptitle("Q Analysis per Transition")

        if self.in_notebook:
            plt.show()

    def on_epoch_end(self, epoch, state, qs, costs, terminals):
        if not self.in_notebook:
            # Activate the relevant figure
            plt.figure(self.fig.number)

        # Plot the state
        for wins in np.where(costs.ravel()[-200:] == 0)[0]:
            self.ax.axvline(wins, color="yellow")
        for channel in range(len(state[0, :])):
            self.ax.plot(channel[-200:, channel, -1], label=f"Channel {channel}")
        for terms in np.where(terminals.ravel()[-200:])[0]:
            self.ax.axvline(terms, color="red")

        self.ax.plot(qs.ravel()[-200:], label="Q")
        self.ax.plot(costs.ravel()[-200:], label="Costs")
        self.ax.axhline(
            min(qs.ravel()[-200:]), linestyle="dotted", color="gray", alpha=0.5
        )
        plt.legend()

        if self.in_notebook:
            self.fig.canvas.draw()
        else:
            # This is what makes it live
            # If you get
            #   AttributeError: type object 'FigureCanvasBase'
            #     has no attribute 'start_event_loop_default'
            # you are in a notebook and notebook recognition failed!
            plt.pause(0.01)
        if self.filepath is not None:
            self.fig.savefig(
                self.filepath, dpi=self.fig.dpi, bbox_inches="tight", pad_inches=0
            )
