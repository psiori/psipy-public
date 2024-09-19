# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plotting for the prioritization ablation studies.

This code is a stripped down version of what can be plotted. See the git history
for this file for other (deprecated) plots.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psipy.rl.io.batch import Episode

FONTSIZE = 20  # 26
plt.rcParams.update({"font.size": FONTSIZE})


def cost_func(states):
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    cost = np.ones(len(states))
    cost[states[..., 3] > 0.98] = 0

    if len(states.shape) == 1:
        return cost[0]
    return cost


def get_bounds(df):  # List of pd.Series
    df.columns = list(range(len(df.columns)))
    cols = df.columns
    df["Mean"] = df[cols].mean(axis=1)
    df["var"] = df[cols].var(axis=1)
    df["Std+"] = df["Mean"] + df.std(axis=1)
    df["Std-"] = df["Mean"] - df.std(axis=1)
    return df


def get_ablation_cost(test_folders):
    """Get the cost over time per trial, plus bounds."""
    costs = {}
    for folder in test_folders:
        this_cost = []
        for ep in sorted(
            glob.glob(os.path.join(folder, "collected_episodes/*.h5")),
            key=os.path.getmtime,
        ):
            episode = Episode.from_hdf5(ep)
            this_cost.append(np.mean(cost_func(episode.observations)))

        # batch = Batch.from_hdf5(os.path.join(folder, "collected_episodes"))
        # batch.compute_costs(cost_func)
        costs[folder] = this_cost  # batch._costs.ravel()
    df = pd.DataFrame.from_dict(costs, orient="index")
    df = df.transpose()
    df = get_bounds(df)
    return df


def plot_learning_curve(ax, ablation_costs, ablations, colors):
    for i, costs in enumerate(ablation_costs):
        costs = costs.rolling(window=50).mean()
        ax.errorbar(
            range(len(costs["Mean"])),
            costs["Mean"],
            yerr=costs["var"],
            label=ablations[i],
            color=colors[i],
            errorevery=25,
            capsize=5,
        )
    ax.legend()
    # ax.set_title("Average Cost per Ablation over Time*")
    ax.set_xlabel("Episode")
    ax.grid(axis="y")
    ax.set_ylabel("Average Goal Cost")


if __name__ == "__main__":

    experiment_path = "PrioritizationSwingupMP"

    def collect_folder_paths(test_name: str, num_tests: int):
        """Get the paths for num_tests folders for the given test."""
        return [
            f"{experiment_path}/{test_name}-test{i}" for i in range(1, num_tests + 1)
        ]

    # Number of trials (repeats) of the same experiment per ablation
    num_repeats = 10

    # Collect data folder paths for loading
    base_data = collect_folder_paths("Base", num_repeats)
    prop_data = collect_folder_paths("PrioProp", num_repeats)
    rank_data = collect_folder_paths("PrioRank", num_repeats)

    # Create handy placeholders for common plotting kwargs
    ablation_labels = ["Base", "Prop", "Rank"]
    base_ablation_kwargs = {"ablation": "Base", "color": "red"}
    prop_ablation_kwargs = {"ablation": "Prop", "color": "green"}
    rank_ablation_kwargs = {"ablation": "Rank", "color": "blue"}

    figure, ax = plt.subplots(figsize=(20, 9))

    plot_learning_curve(
        ax,
        [
            get_ablation_cost(base_data),
            get_ablation_cost(prop_data),
            get_ablation_cost(rank_data),
        ],
        ablation_labels,
        ["red", "green", "blue"],
    )

    plt.savefig("prioritization.eps", bbox_inches="tight")
    plt.show()
