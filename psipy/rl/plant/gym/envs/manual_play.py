#!/usr/bin/env python

"""Use the keyboard to explore behavior of gyms.

Pass environment name as a command-line argument, for example:
    python manual_play.py SpaceInvadersNoFrameskip-v4

From: github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

Note that in order to play some environments, they need to be temporarily altered to do
meaningful actions with just inputs in the range 0-9, since this code does not respect
the action definition of the action class. For instance, cartpole needs to transform
[0, 1, 2] -> [-10, 0, 10].
"""

import sys
import time

import gym
from gym.envs.registration import register


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xFF0D:
        human_wants_restart = True
    if key == 32:
        human_sets_pause = not human_sets_pause
    a = int(key - ord("0"))
    if a < 0 or a >= ACTIONS:
        print("Invalid action...")
        return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord("0"))
    if a < 0 or a >= ACTIONS:
        return
    if human_agent_action == a:
        human_agent_action = DEFAULT_ACTION


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        print(obser)
        if r != 0:
            print("Cost: %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open is False:
            return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(SPEED)
        time.sleep(SPEED)
    print("End Timestep: %i; Cost: %0.2f" % (total_timesteps, total_reward))


if __name__ == "__main__":
    register(
        id="CartPoleUnbounded-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleUnboundedEnv",
        max_episode_steps=500,
        reward_threshold=195.0,
    )
    register(
        id="CartPoleSwingUp-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleSwingUpEnv",
        max_episode_steps=1000,
        reward_threshold=195.0,
    )
    register(
        id="CartPoleBalance-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleBalanceEnv",
        max_episode_steps=1000,
        reward_threshold=195.0,
    )

    env = gym.make("CartPoleUnbounded-v0" if len(sys.argv) < 2 else sys.argv[1])

    def cost_func1(states):
        from psipy.rl.control.nfq import tanh2
        import numpy as np
        cost = np.zeros(len(states))
        cost[states[..., 5] < -.75] += (1 / 200) / 2
        cost += tanh2(states[..., 5] - 1, C=(1 / 200 )/ 2, mu=0.05)

        return cost


    env.env._cost_func = cost_func1
    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")
    ACTIONS = env.action_space.n
    # Choose actions every SKIP_CONTROL steps
    SKIP_CONTROL = 0
    DEFAULT_ACTION = 1
    SPEED = 0.055  # how fast the sim refreshes; lower is faster
    human_agent_action = DEFAULT_ACTION
    human_wants_restart = False
    human_sets_pause = False

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3... to take actions 1 2 3...")
    print(f"No keys pressed is taking action {DEFAULT_ACTION}")
    while 1:
        window_still_open = rollout(env)
        if window_still_open is False:
            break
