from datetime import datetime
import os
import shutil

import math
import numpy as np
import tensorflow as tf

import matplotlib

matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt

from psipy.rl import Loop
from psipy.rl.plant.gym import BasicCartPoleSwayPlant
from psipy.rl.control.nfq import NFQ
from psipy.rl.control.memory import Memory


def make_model(num_inputs, num_outputs):
    inp = tf.keras.layers.Input((1, num_inputs))
    net = tf.keras.layers.Flatten()(inp)
    net = tf.keras.layers.Dense(20, activation="tanh")(net)
    net = tf.keras.layers.Dense(20, activation="tanh")(net)
    net = tf.keras.layers.Dense(num_outputs, activation="sigmoid")(net)
    model = tf.keras.Model(inputs=inp, outputs=net)
    model.summary()
    return model


plant = BasicCartPoleSwayPlant(render=True)

memory = Memory()
model = make_model(
    num_inputs=len(plant.state_channels), num_outputs=plant.action_type.num_values
)
ctrl = NFQ(
    model, state_channels=plant.state_type.state_channels, actions=plant.action_type, memory=memory
)

loop = Loop(plant=plant, control=ctrl, store=memory)

min_epsilon = 0.01
decay = 0.75

max_steps = 200
_, plant_rewards, steps = loop.run_episode(max_steps=max_steps)
rewards_means = [np.mean(plant_rewards)]
rewards_sums = [np.sum(plant_rewards)]
rewards_all = plant_rewards

for i in range(1, 201):
    _, plant_rewards, steps = loop.run_episode(max_steps=max_steps)
    rewards_means.append(np.mean(plant_rewards))
    rewards_sums.append(np.sum(plant_rewards))
    rewards_all += plant_rewards
    print(
        f"Episode {i} with {steps[0]} steps (eps: {control._epsilon:1,.2}), "
        f"replaysize: {len(memory)}"
    )

    # print(
    #     f"r: {np.mean(plant_rewards):1,.4f}, "
    #     f"avg(r): {np.mean(rewards_means[-5:]):1,.4f}"
    # )
    # if True:  # np.mean(plant_rewards) < np.mean(rewards_means[-5:]):
    #     print(f"Reduce eps")
    #     new_eps = max(min_epsilon, control._epsilon * decay)
    #     control.set_epsilon(new_eps)
    # # control.set_epsilon(1 / max(1, i - 10))
    # plant._render = False
    # if True:  # i % 5 == 0:

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set_xlabel("Episode")
    #     ax.plot(rewards_means, label="cost", color="blue")
    #     plt.title("EpisodeCosts")
    #     plt.savefig(os.path.join(fig_path, "episode_costs.png"))
    #     plt.close()

    #     # control.controller.save(dirpath=os.path.join(fig_path, "model"))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set_xlabel("Episode")
    #     ax.plot(rewards_sums, label="cost", color="blue")
    #     plt.title("EpisodeCosts")
    #     plt.savefig(os.path.join(fig_path, "episode_costs_sum.png"))
    #     plt.close()

    #     plt.plot(rewards_all)
    #     plt.title("AllCosts")
    #     plt.xlabel("Transition")
    #     plt.ylabel("cost")
    #     plt.savefig(os.path.join(fig_path, "all_costs.png"))
    #     plt.close()

    #     plant._render = True
    #     trainer.fit(
    #         replay_buffer,
    #         epochs=20,
    #         epochs_per_dp=25,
    #         i=i,
    #         # callbacks=[tensorboard_callback],
    #         minibatch_size=-1,
    #         fig_path=fig_path,
    #     )
    #     # control.controller.save(dirpath=os.path.join(fig_path, "model"))
    #     # control.set_epsilon(1 / max(1, i - 10))
    #     cmd = (
    #         f"ffmpeg -y -i '{os.path.join(fig_path, '%*.png')}' "
    #         f"-vcodec libx264 -crf 25 -pix_fmt yuv420p "
    #         f"{os.path.join(fig_path, 'evolution.mp4')}"
    #     )
    #     # os.system(cmd)

    #     # run one episode with 'optimal' policy.
    #     # eps_saved = control._epsilon
    #     # control.set_epsilon(0.)

    # # if i > 10 and i % 10 == 1:
    # #     control.set_epsilon(eps_saved)
