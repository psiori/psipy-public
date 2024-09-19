# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import io
import sys
from typing import BinaryIO, Callable, Dict, Optional

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa E402
from mpl_toolkits.mplot3d import Axes3D  # noqa E402


def plot_scatter(embedding: np.ndarray) -> plt.figure:
    """Create a 2 or 3D scatter plot, automatically falling to PCA for nD data."""
    kwargs: Dict = dict(xticklabels=[], yticklabels=[])
    three_d = embedding.shape[1] >= 3
    if embedding.shape[1] <= 3:  # 2D or 3D code layer
        kwargs = dict(ylim=(0, 1), xlim=(0, 1), **kwargs)
        if three_d:
            kwargs = dict(projection="3d", zlim=(0, 1))
    else:  # PCA
        kwargs = dict(projection="3d", title="PCA", **kwargs)
        embedding = PCA(n_components=3).fit_transform(embedding)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, **kwargs)
    ax.scatter(*embedding.T)
    return fig


def fig_to_buf(fig: plt.figure) -> BinaryIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


class TensorboardScatterCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_dir: str,
        data: np.ndarray,
        plot_func: Callable = plot_scatter,
        name: Optional[str] = None,
        every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self._log_dir = log_dir
        self._plot_func = plot_func
        self._data = data
        self._name = name or "scatter"
        self._epoch = 0
        self._every_n_epochs = every_n_epochs
        self._built = False

    def set_model(self, model):
        super().set_model(model)
        self._sess = tf.compat.v1.keras.backend.get_session()
        self._embed = tf.keras.models.Sequential(
            model.layers[: len(model.layers) // 2 + 1]
        )
        if not self._built:
            self._ph = tf.compat.v1.placeholder(tf.string)
            image = tf.image.decode_png(self._ph, channels=4)
            image = tf.expand_dims(image, 0)  # Convert to "batch".
            self._summary_op = tf.compat.v1.summary.image(
                self._name, image, max_outputs=1
            )
            self._writer = tf.compat.v1.summary.FileWriter(self._log_dir)
            self._built = True
        else:
            self._writer.reopen()

    def on_epoch_end(self, epoch: int, logs: Dict):
        if self._epoch % self._every_n_epochs == 0:
            self.plot()
        self._epoch += 1

    def plot(self):
        code = self._embed.predict(self._data)
        buff = fig_to_buf(self._plot_func(code)).getvalue()
        summary = self._sess.run(self._summary_op, {self._ph: buff})
        self._writer.add_summary(summary, self._epoch)
        self._writer.flush()

    def on_train_end(self, _):
        if self._epoch % self._every_n_epochs != 0:
            self.plot()
        self._writer.close()
