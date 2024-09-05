# Multi-Armed Bandits

Here is a collection of multi armed bandit optimizers for use in problems such as A/B testing and reinforcement learning.

|                    | Stationary | Nonstationary | Distribution Dependent |
|--------------------|------------|---------------|------------------------|
| Epsilon Greedy     | x          | x             |                        |
| UCB1               | x          | x             |                        |
| Sliding Window UCB |            | x             |                        |
| Softmax            | x          | x             |                        |
| Thompson Sampling  | x          |               | x                      |



Note: the implementation and interface is based off of https://github.com/lilianweng/multi-armed-bandit.

#### Sources
* https://www.youtube.com/watch?v=VrFZCGCwzVc (5:12)
* https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
* https://en.wikipedia.org/wiki/Multi-armed_bandit
* https://en.wikipedia.org/wiki/Thompson_sampling
* https://towardsdatascience.com/beta-distribution-intuition-examples-and-derivation-cf00f4db57af
* https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-softmax-algorithm-e1fa4cb0c422
* https://www.cs.mcgill.ca/~vkules/bandits.pdf