# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plotting ofr Q scaling ablation studies."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psipy.rl.control.nfq import tanh2
from psipy.rl.io.batch import Batch


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


root_dir = "H_q_scaling_experiments"
repeats = 10
cs = ["100", "1", "0.1", "0.01", "0.001", "0.0001"]
ablations = ["none", "hafner", "Riedmiller", "Both"]
q_curve_colors = ["tab:orange", "tab:green", "tab:red"]
q_curve_labels = ["Min Q", "Avg Q", "Max Q"]
AB_colors = ["C0", "black", "tab:orange", "tab:red"]
AB_labels = ["A", "B", "Min Q", "Max Q"]


def costfunc(states):
    return tanh2(states[..., 2], C=1, mu=0.05)


def read_num_solves(path):
    with open(path) as reader:
        num_solves = int(reader.read())
    return num_solves


def get_bounds(df):  # List of pd.Series
    df["Mean"] = df.mean(axis=1)
    df["Std+"] = df["Mean"] + df.std(axis=1)
    df["Std-"] = df["Mean"] - df.std(axis=1)
    return df


def plot_q_level(axes, level: str, initial, last):
    level_dir = join(root_dir, level)

    for i, (ax, ablation) in enumerate(zip(axes, ablations)):
        ablation_dir = join(level_dir, ablation)
        histories = [
            pd.read_csv(join(ablation_dir, str(run), "q_history.csv"))
            for run in range(repeats)
        ]
        min_q = [history["min_q"] for history in histories]
        max_q = [history["max_q"] for history in histories]
        avg_q = [history["avg_q"] for history in histories]
        min_q = get_bounds(pd.concat(min_q, axis=1))
        max_q = get_bounds(pd.concat(max_q, axis=1))
        avg_q = get_bounds(pd.concat(avg_q, axis=1))

        for curve, color, label in zip(
            [min_q, avg_q, max_q], q_curve_colors, q_curve_labels
        ):
            ax.plot(curve["Mean"], "-", color=color, label=label)
            ax.plot(curve["Std+"], "-", color=color, alpha=0.5)
            ax.plot(curve["Std-"], "-", color=color, alpha=0.5)

            ax.fill_between(
                range(len(curve)), curve["Std+"], curve["Std-"], color=color, alpha=0.05
            )

        if initial:
            ax.legend()
            ax.set_title(ablation.capitalize())

        if last:
            ax.set_xlabel("Epoch")

        if i == 0:
            ax.set_ylabel(f"C={level}")


def plot_success_failure_bars(ax, level: str, initial):
    level_dir = join(root_dir, level)
    x = np.arange(len(ablations))  # the label locations

    wins = []
    losses = []
    for i, ablation in enumerate(ablations):
        ablation_dir = join(level_dir, ablation)
        num_solves = [
            read_num_solves(join(ablation_dir, str(run), "numsolves"))
            for run in range(repeats)
        ]
        num_fails = [10 - solve for solve in num_solves]
        # Compute mean and std using 100 runs instead of the 10 np.mean/std would think happened
        win_mean = np.mean(num_solves)
        lose_mean = np.mean(num_fails)

        wins.append(win_mean)
        losses.append(lose_mean)

    width = 0.35  # the width of the bars
    rects2 = ax.bar(x - width / 2, losses, width, label="Failure", color="red",)
    rects1 = ax.bar(x + width / 2, wins, width, label="Success", color="green",)
    # autolabel(ax,rects1)
    # autolabel(ax,rects2)
    # ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([ablation.capitalize() for ablation in ablations])
    ax.set_title(f"C={level}")

    if initial:
        ax.set_ylabel("Average Success (green) / Failure (red)")


def plot_quality_of_policy(ax, level: str, initial):
    level_dir = join(root_dir, level)

    ablation_costs = {}
    for ablation in ablations:
        ablation_dir = join(level_dir, ablation)
        batches = [
            Batch.from_hdf5(join(ablation_dir, str(i), "eval")) for i in range(repeats)
        ]
        costs = []
        for batch in batches:
            batch.compute_costs(costfunc)
            costs.extend(
                batch.set_minibatch_size(-1).costs_terminals[0][0].ravel().tolist()
            )
        ablation_costs[ablation] = costs

    ax.boxplot([v for v in ablation_costs.values()])
    ax.set_xticklabels([a.capitalize() for a in ablations])
    ax.set_title(f"C={level}")
    if initial:
        ax.set_ylabel("Cost")


def plot_AB_minmax(axes, level, initial):
    level_dir = join(root_dir, level)

    for i, (ax, ablation) in enumerate(zip(axes, ablations)):
        ablation_dir = join(level_dir, ablation)

        dfs = [
            pd.read_csv(join(ablation_dir, f"q_scaling-{level}-{i}-{ablation}.csv"))
            for i in range(10)
        ]
        A = [df["A"] for df in dfs]
        B = [df["B"] for df in dfs]
        minQ = [df["outmin"] for df in dfs]
        maxQ = [df["outmax"] for df in dfs]

        A = get_bounds(pd.concat(A, axis=1))
        B = get_bounds(pd.concat(B, axis=1))
        minQ = get_bounds(pd.concat(minQ, axis=1))
        maxQ = get_bounds(pd.concat(maxQ, axis=1))

        cut = 2
        curves = [A, B, minQ, maxQ]
        for curve, color, label in zip(curves[cut:], AB_colors[cut:], AB_labels[cut:]):
            if not initial:
                label = None
            ax.scatter(
                range(len(curve)), curve["Mean"], color=color, label=label, s=0.75
            )
            ax.plot(curve["Std+"], "-", color=color, alpha=0.25)
            ax.plot(curve["Std-"], "-", color=color, alpha=0.25)
            ax.fill_between(
                range(len(curve)),
                curve["Std+"],
                curve["Std-"],
                color=color,
                alpha=0.025,
            )
            if initial:
                ax.legend()
                ax.set_title(ablation.capitalize())

        ax2 = ax.twinx()
        if ablation in ["Riedmiller", "Both"]:
            for curve, color, label in zip(
                curves[:cut], AB_colors[:cut], AB_labels[:cut]
            ):
                if not initial:
                    label = None
                ax2.scatter(
                    range(len(curve)), curve["Mean"], color=color, label=label, s=0.75
                )
                ax2.plot(curve["Std+"], "-", color=color, alpha=0.25)
                ax2.plot(curve["Std-"], "-", color=color, alpha=0.25)
                ax2.fill_between(
                    range(len(curve)),
                    curve["Std+"],
                    curve["Std-"],
                    color=color,
                    alpha=0.025,
                )

                if initial:
                    ax2.legend()

        if i == 0:
            ax.set_ylabel(f"C={level}")


if __name__ == "__main__":

    figure, axes = plt.subplots(nrows=2, ncols=len(cs), figsize=(20, 9))
    for i, c in enumerate(cs):
        plot_success_failure_bars(axes[0, i], c, initial=i == 0)
        plot_quality_of_policy(axes[1, i], c, initial=i == 0)
    text = (
        "The following cost function was used with varying levels of max cost (C): tanh2(angle, C=c, mu=0.05). "
        "C's tested were [100, 1, 0.1, 0.01, 0.001, 0.0001].\n"
        "All networks were trained with the same regime (40 DP steps, 20 epochs per step, γ=.99, no exploration); the only difference being how the network scaled for C.\n"
        "Four methods were ablated: no scaling, scaling according to Hafner, scaling according to Riedmiller, and both scaling methods together.\n"
        "Hafner: subtract the minimum Q, slide Q up by .05 and clip to range [.05, .95].\n"
        "Riedmiller: scale Q by a bijective linear function whose parameters update during training.\n"
        "Experiments were averaged over independent 10 runs."
    )
    plt.text(-9.5, -0.35, text, ha="center")
    plt.suptitle("Quality of Learned Policies per Ablation and Cost Level")
    plt.show()
    figure, axes = plt.subplots(nrows=len(cs), ncols=len(ablations), figsize=(20, 9))
    for i, c in enumerate(cs):
        plot_q_level(axes[i, ...], c, initial=i == 0, last=i == len(cs) - 1)
    plt.suptitle("Q Evolution During Training")
    plt.tight_layout()
    plt.show()
    figure, axes = plt.subplots(nrows=len(cs), ncols=len(ablations), figsize=(20, 9))
    for i, c in enumerate(cs):
        plot_AB_minmax(axes[i, ...], c, initial=i == 0)
    plt.suptitle("Network Scaling Parameters During Training")
    plt.tight_layout()
    plt.show()
