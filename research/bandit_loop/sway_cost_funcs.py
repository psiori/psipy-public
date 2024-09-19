"""Find a best controller for CartpoleSwingup with the use of a Multi Armed Bandit."""
from functools import partial
from typing import List

import numpy as np

from psipy.rl.control.nfq import tanh2
from research.bandit_loop.cost_functions import C9, C99, C999


def _cost_func_stepped(states, C):
    cos = states[..., 3]
    cost = np.ones(len(states)) * C
    cost[cos < 0.9] -= C / 2 / 5
    cost[cos < 0.5] -= C / 2 / 5
    cost[cos < 0] -= C / 2 / 5
    cost[cos < -0.5] -= C / 2 / 5
    cost[cos < -0.9] = 0

    pos = states[..., 0]
    cost[np.abs(pos) > 2] += C / 2 / 4
    cost[np.abs(pos) > 1.5] += C / 2 / 4
    cost[np.abs(pos) > 1] += C / 2 / 4
    cost[np.abs(pos) > 0.5] += C / 2 / 4

    if len(states.shape) == 1:
        return cost[0]
    return cost


def _cost_func_onoff(states, C):
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    cost = np.ones(len(states)) * C
    cos = states[...,3]
    pos = states[...,0]
    cost[(cos < -.98) & (np.abs(pos) < .1)] = 0
    # cost[states[..., 3] < -0.98] = 0
    # pos = np.ones(len(states)) * C / 2
    # pos[np.abs(states[..., 0]) < 0.1] = 0
    # cost += pos
    if len(states.shape) == 1:
        return cost[0]
    return cost


def _cost_func_tanh2(states, C):
    cost = tanh2(states[..., 3] + 1, C=C / (3 / 4), mu=0.95)
    pos_cost = tanh2(states[..., 0], C=C / 4, mu=0.1)
    return cost + pos_cost


def _cost_func_direct(states, C):
    cos = -states[..., 3]
    cost = (cos - 1) / 2
    pos_cost = np.abs(states[..., 0])
    pos_cost /= 2.4
    return (np.abs(cost) + pos_cost) * 0.5 * C  # .5 because cos + pos = 2 not 1


def bandit_cost(states: List[np.ndarray]):
    states = np.array(states)
    cos = -states[..., 3]
    pos = states[..., 0]
    cos_cost = np.abs(((cos - 1) / 2))
    pos_cost = np.abs(pos)
    pos_cost /= 2.4
    return cos_cost + pos_cost

def bandit_cost_4batch(states):
    cos = -states[..., 3]
    pos = states[..., 0]
    cos_cost = np.abs(((cos - 1) / 2))
    pos_cost = np.abs(pos)
    pos_cost /= 2.4
    return cos_cost + pos_cost


# import matplotlib.pyplot as plt
# angles = np.linspace(-1,1,100)
# pos = np.linspace(-2.4, 2.4, 100)
# import seaborn as sns
# states = []
# for i in range(100):
#     for z in range(100):
#         states.append(np.array([pos[i], 0, 0, angles[z]]))
# cost = _cost_func_stepped(np.array(states), C=.5)
# sns.heatmap(cost.reshape((100,100)))
# plt.show()
# print(_cost_func_tanh2(np.array([np.array([0.0, 0, 0, -1]),np.array([0, 0, 0, -1])])))
# exit()


# Cost functions with equal expected future costs based off gamma
cost_func_direct9 = partial(_cost_func_direct, C=C9)
cost_func_direct99 = partial(_cost_func_direct, C=C99)
cost_func_direct999 = partial(_cost_func_direct, C=C999)

cost_func_onoff1 = partial(_cost_func_onoff, C=1)
cost_func_onoff9 = partial(_cost_func_onoff, C=C9)
cost_func_onoff99 = partial(_cost_func_onoff, C=C99)
cost_func_onoff999 = partial(_cost_func_onoff, C=C999)

cost_func_stepped9 = partial(_cost_func_stepped, C=C9)
cost_func_stepped99 = partial(_cost_func_stepped, C=C99)
cost_func_stepped999 = partial(_cost_func_stepped, C=C999)

cost_func_tanh29 = partial(_cost_func_tanh2, C=C9)
cost_func_tanh299 = partial(_cost_func_tanh2, C=C99)
cost_func_tanh2999 = partial(_cost_func_tanh2, C=C999)
