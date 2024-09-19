import glob

import numpy as np

from psipy.rl.control.nfq import tanh2
from psipy.rl.io.batch import Batch, Episode

# from psipy.rl.plant.gym.cartpole_plants import CartPoleSwayState
from psipy.rl.io.sart import SARTReader
from psipy.rl.plant.cartpole import CartPoleState as CartPoleSwayState
from matplotlib import pyplot as plt

from psipy.rl.plant.swingup_plant import SwingupPlant, SwingupState

# folder = "run2"
folder = "swingup4-retrain/"

STATE_CHANNELS = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_velocity",
    "direction_ACT",
)
EPS_STEPS=200
THETA_CHANNEL_IDX = STATE_CHANNELS.index("pole_angle")
def costfunc(states: np.ndarray) -> np.ndarray:
    # Set the middle position to be 0 for the position costs
    position = states[:,0]
    theta = states[:, THETA_CHANNEL_IDX]
    theta_speed = states[:, THETA_CHANNEL_IDX + 1]

    c = 1/EPS_STEPS - .003
    costs = tanh2(theta, C=c, mu=.5)
    costs[np.abs(theta_speed) > .5] += .003
    # costs[position < SwingupPlant.LEFT_SIDE+20] = 1
    # costs[position > SwingupPlant.RIGHT_SIDE-20] = 1

    return costs #+ cost_x



def create_fake_episodes(folder: str, lookback: int):
    kwargs = dict(lookback=lookback)

    # According to NFQ Tricks, add goal states with all 3 actions
    # These are added 200 times
    o = np.zeros((200, 5))  # 5th channel is 0-action enrichment
    o[:, 0] = 450
    t = np.zeros(200)
    a = np.ones(200)  # 1 == do nothing
    c = np.zeros(200)
    yield Episode(o, a, t, c, **kwargs)

    kwargs = dict(lookback=lookback)
    for path in glob.glob(f"{folder}/*.h5"):
        try:
            with SARTReader(path) as reader:
                o, a, t, c = reader.load_full_episode(
                    state_channels=SwingupState.channels(),
                    action_channels=("direction_index",),
                )
            a_ = a.copy()
            a_[a == 0] = 2
            a_[a == 2] = 0
            o[:, 0] -= SwingupPlant.CENTER
            o = -1 * o
            o[:, 0] += SwingupPlant.CENTER
            yield Episode(o, a_, t, c, **kwargs)
        except (KeyError, OSError):
            continue


state_channels = SwingupState.channels()
lookback = 2
batch = Batch.from_hdf5(folder, lookback=lookback)
# batch.append(create_fake_episodes(folder, lookback))
# Update cost function
batch.compute_costs(costfunc)

states, costs = batch.states_costs.set_minibatch_size(-1)[0]
print(states.shape)
from mpl_toolkits.mplot3d import axes3d, Axes3D

figure, ax = plt.subplots(ncols=2, figsize=(20, 9), subplot_kw={"projection": "3d"})
ax[0].scatter(
    states[..., 0, lookback-1], states[..., 2, lookback-1], costs, s=1, c=states[..., 1, lookback-1].flatten()
)
ax[0].set_xlabel("x")
ax[0].set_ylabel("theta")
ax[0].set_zlabel("cost")
ax[0].set_title("X/Theta")

ax[1].scatter(
    states[..., 2, 0], states[..., 3, 0], costs, s=1, c=states[..., 1, 0].flatten()
)
ax[1].set_xlabel("theta")
ax[1].set_ylabel("theta_speed")
ax[1].set_zlabel("cost")
ax[1].set_title("Angle/AngleDot")
plt.suptitle("Batch Costs")
plt.show()
