# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Create a simple visualization of data stored in SART files.

Currently supported plots:
    Any number of state channels plotted on one plot, with start/stop trimming
"""

__all__ = ["plot_observations"]

from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from psipy.rl.io.sart import SARTReader
from psipy.rl.core.plant import State


def plot_observations(
    sart_path: str,
    state_type: State,
    channels: Optional[List[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    filename: str = "",
    show: bool = True,
    **subplot_kwargs,
) -> None:
    """Plot observations from a SART file.

    Args:
        state_type: state type object of the state the sart writer recorded
        channels: a subset of channels that you want to plot
        start: timepoint to start plotting
        end: timepoint to end plotting
        filename: if provided, will save the figure. Without extension!
        show: if False, will not display figure on screen
        subplot_kwargs: for example, figsize
    """
    with SARTReader(sart_path) as reader:
        sart = reader.load_full_episode(state_channels=channels)
    obs = sart[0]

    state = dict(zip(state_type.channels(), obs.T))
    tag2sem: Optional[Dict] = None
    semantic_channels = state_type.semantic_channels()
    if semantic_channels != state_type.channels():  # then semantics exist
        tag2sem = dict(zip(state_type.channels(), semantic_channels))

    # TODO: Maybe could allow passing a groupby function which allows
    # to plot multiple channels into common subplots
    fig, ax = plt.subplots(nrows=len(state), **subplot_kwargs)
    for i, (key, value) in enumerate(state.items()):
        if tag2sem is not None:
            ax[i].set_title(f"{tag2sem[key]}/{key}")
        else:
            ax[i].set_title(f"{key}")
        ax[i].plot(value[start:end])
    plt.tight_layout()

    # Generate a title and filename to mention it is zoomed
    zoom = ""
    if start is not None or end is not None:
        zoom = f" - Zoomed {start}-{end}"
        if filename != "":
            filename = f"{filename}-{start}-{end}"

    # Set title and plot
    plt.suptitle(
        f"Observations for {state_type.__class__}{zoom} "
        f"({datetime.now().strftime('%d.%m.%y')})",
        y=1.02,
    )

    if filename != "":
        plt.savefig(f"{filename}.png", bbox_inches="tight")

    if show:
        plt.show()


if __name__ == "__main__":
    import glob
    import shutil

    from psipy.rl.control.controller import ContinuousRandomActionController
    from psipy.rl.loop import Loop
    from psipy.rl.plant.tests.mocks import MockAction, MockPlant, MockState

    loop = Loop(
        MockPlant(),
        ContinuousRandomActionController(MockState.channels(), MockAction),
        "VisualizeLoop",
        logdir="deleteme",
    )
    loop.run(1, 20)

    files = glob.glob("deleteme/*.h5")

    for i, filepath in enumerate(files):
        plot_observations(
            filepath,
            MockState,
            figsize=(10, 20),
            filename=f"visualize-{i}",
            show=True,
        )

    shutil.rmtree("deleteme")
    try:
        shutil.rmtree("default_log")
    except FileNotFoundError:
        pass
