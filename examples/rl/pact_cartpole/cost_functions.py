from functools import partial
from typing import List

import numpy as np

from psipy.rl.control.nfq import expected_discounted_cost, tanh2

__all__ = [
    "bandit_cost",
    "bandit_cost_4batch",
    "C9",
    "C99",
    "C999",
    "cost_func_cos9",
    "cost_func_cos99",
    "cost_func_cos999",
    "cost_func_onoff1",
    "cost_func_onoff9",
    "cost_func_onoff99",
    "cost_func_onoff999",
    "cost_func_shaped9",
    "cost_func_shaped99",
    "cost_func_shaped999",
    "cost_func_stepped9",
    "cost_func_stepped99",
    "cost_func_stepped999",
    "cost_func_tanh29",
    "cost_func_tanh299",
    "cost_func_tanh2999",
]

C9 = expected_discounted_cost(200, 0.9)
C99 = expected_discounted_cost(200, 0.99)
C999 = expected_discounted_cost(200, 0.999)


def _cost_func_stepped(states, C):
    cos = states[..., 3]
    cost = np.ones(len(states)) * C
    cost[cos > -0.9] -= C / 5
    cost[cos > -0.5] -= C / 5
    cost[cos > 0] -= C / 5
    cost[cos > 0.5] -= C / 5
    cost[cos > 0.9] = 0

    if len(states.shape) == 1:
        return cost[0]
    return cost


def _cost_func_onoff(states, C):
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    cost = np.ones(len(states)) * C
    cost[states[..., 3] > 0.98] = 0

    if len(states.shape) == 1:
        return cost[0]
    return cost


def _cost_func_tanh2(states, C):
    cost = tanh2(states[..., 3] - 1, C=C, mu=0.95)
    return cost


def _cost_func_cos(states, C):
    cos = states[..., 3]
    cost = ((-cos + 1) / 2) * C
    return cost


def _shaped_cost_function(states, C):
    cos = states[..., 3]
    rate = np.abs(states[..., 4] / 100 / 200)  # make it not add up to 1 so quickly
    cost = ((-cos + 1) / 2) * C
    rate_cost = np.zeros(len(states))
    if np.any(cos > 0):
        rate_cost[cos > 0] = rate[cos > 0]
    cost = cost + rate_cost
    if len(states.shape) == 1:
        return cost[0]
    return cost


def bandit_cost(states: List[np.ndarray]):
    cos = np.array(states)[..., 3]
    pole_speed = np.array(states)[..., 4]
    rate = np.abs(pole_speed / 100)  # make it not overcome the angle cost
    cost = (-cos + 1) / 2
    rate_cost = np.zeros(len(states))
    if np.any(cos > 0):
        rate_cost[cos > 0] = rate[cos > 0]
    cost = cost + rate_cost
    return cost


def bandit_cost_4batch(states: np.ndarray):
    cos = states[..., 3]
    pole_speed = states[..., 4]
    rate = np.abs(pole_speed / 100)  # make it not overcome the angle cost
    cost = (-cos + 1) / 2
    rate_cost = np.zeros(len(states))
    if np.any(cos > 0):
        rate_cost[cos > 0] = rate[cos > 0]
    cost = cost + rate_cost
    return cost


# Cost functions with equal expected future costs based off gamma
cost_func_cos9 = partial(_cost_func_cos, C=C9)
cost_func_cos99 = partial(_cost_func_cos, C=C99)
cost_func_cos999 = partial(_cost_func_cos, C=C999)

cost_func_onoff1 = partial(_cost_func_onoff, C=1)
cost_func_onoff9 = partial(_cost_func_onoff, C=C9)
cost_func_onoff99 = partial(_cost_func_onoff, C=C99)
cost_func_onoff999 = partial(_cost_func_onoff, C=C999)

cost_func_shaped9 = partial(_shaped_cost_function, C=C9)
cost_func_shaped99 = partial(_shaped_cost_function, C=C99)
cost_func_shaped999 = partial(_shaped_cost_function, C=C999)

cost_func_stepped9 = partial(_cost_func_stepped, C=C9)
cost_func_stepped99 = partial(_cost_func_stepped, C=C99)
cost_func_stepped999 = partial(_cost_func_stepped, C=C999)

cost_func_tanh29 = partial(_cost_func_tanh2, C=C9)
cost_func_tanh299 = partial(_cost_func_tanh2, C=C99)
cost_func_tanh2999 = partial(_cost_func_tanh2, C=C999)


if __name__ == "__main__":
    print(cost_func_cos99(np.array([0, 0, 0, 0, 0, 0])))
    print(cost_func_cos999(np.array([0, 0, 0, 0, 0, 0])))
    from typing import List

    import matplotlib.pyplot as plt
    import numpy as np

    FONTSIZE = 18
    plt.rcParams.update({"font.size": FONTSIZE})
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 1000)
    cosine = np.cos(x)
    states = np.array([np.zeros(1000), np.zeros(1000), np.zeros(1000), cosine]).T
    coscost = _cost_func_cos(states, C=1)
    stepcost = _cost_func_stepped(states, C=1)
    sparsecost = _cost_func_onoff(states, C=1)
    tanhcost = _cost_func_tanh2(states, C=1)
    figure, ax = plt.subplots(figsize=(20, 9))
    ax2 = ax.twinx()
    p1 = ax2.plot(x, cosine, label="cos(θ)", color="black", ls="--")
    p2 = ax.plot(x, coscost, label="Cosine cost")
    p4 = ax.plot(x, sparsecost, label="Sparse cost")
    p3 = ax.plot(x, stepcost, label="Stepped cost")
    p5 = ax.plot(x, tanhcost, label="Tanh2 cost")
    lns = p1 + p2 + p3 + p4 + p5
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="center right")
    ax.set_title("Cost Functions")
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Cost")
    ax2.set_ylabel("cos(θ)")
    # plt.savefig("cost_functions.eps")
    plt.show()
