# import numpy as np

# from baselines.ddpg.memory import Memory as BaselinesMemory


# class Memory(BaselinesMemory):
#     def append(self, transition: Transition):
#         obs0 = np.asarray(transition.x.values)
#         action = transition.action
#         reward = transition.reward
#         obs1 = np.asarray(transition.x_.values)
#         terminal1 = transition.terminal
#         return BaselinesMemory.append(self, obs0, action, reward, obs1, terminal1)
