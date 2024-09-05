from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf

from psipy.core.notebook_tools import is_notebook


class MultiLossCallback(tf.keras.callbacks.Callback):
    """A callback that plots the loss of n many algorithms of any type.

    To use:

        1. Generate a subplot figure with the subplot dimensions desired.
        2. Flatten the axes if using both rows and columns.
        3. Create an instance of this class for each model, each with a different
           axis from the figure.
        4. Provide the figure number so that Matplotlib knows where to put the data.
        5. Pass the callbacks into their respective model's .fit() function.

    """

    def __init__(
        self,
        axis: plt.axis,
        fig_num: int,
        title: Optional[str] = None,
        filepath: Optional[str] = None,
    ):
        """Keras callback for live loss plotting across n asynchronous fittings."""
        if is_notebook():
            raise NotImplementedError()
        self.filepath = filepath
        self.in_notebook = is_notebook()
        self.axis = axis
        self.fignum = fig_num
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel("Loss")
        self.title = title

        self.loss = []

    def on_epoch_end(self, epoch, metrics):
        if not self.in_notebook:
            # Activate the relevant figure
            plt.figure(self.fignum)
        self.loss.append(metrics["loss"])
        self.axis.clear()
        self.axis.plot(self.loss)
        if self.title is not None:
            self.axis.set_title(self.title)
        plt.pause(0.01)

    def __del__(self):
        plt.close(self.fignum)