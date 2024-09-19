# PSIORI Reinforcement Learning Toolbox

**DEPRECATED**: Please refer to rest of psipy documentation.

Toolbox for closed loop control using reinforcement learning, specifically using deep neural network based function approximators trained using batch reinforcement learning approaches.

## Usage

```bash
pip install git@github.com:psiori/DeepBatchRL.git
```

After having done so, it can be used along the following lines:

```python
from psipy.rl import SomePlant, QController, DiscreteExplorer

def main():
    plant = SomePlant()
    agent = QController()
    explorer = DiscreteExplorer(agent)

    def step(state):
        action = explorer.get_action(state)
        return action

    plant.step = step
```

# Top Level Interfaces

## loop

The primary control loop. It manages the interaction between exactly one plant and one controller, following a specific sequential chain of commands and providing checks (e.g. for cycle times) and logging capabilities.

- `run` executes a single episode
- `run_epsiodes` executes multiple episodes

Before each episode the plant and controller are notified that an episode is about to start. Similarly, both are notified when an episode stops. Episode stops are generally determined by the plant (a state is a terminal state), but can also be initiated by some outside factor (e.g. a user manually stopping the episode). Start and stop notifications should be used for preparations (e.g. start up the simulator) and cleanup (e.g. return a physical model to a neutral position).

Within each episode:

1. Notify plant of episode start
2. Notify controller of episode start
3. Get initial state from plant
4. Get action from controller given state
5. Get next state from plant given previous state and action
6. Repeat from 3 if not in terminal state or some other stop condition holds
7. Notify controller of episode end
8. Notify plant of episode end

While performing these steps, all states (including the initial state) and actions are pushed to thread-safe queues which are read and logged to disk in an independent thread.

**The loop (or plant?) is responsible for reliable cycle times.** If a cycle is too fast, it will sleep to make the cycle time deterministic. If a cycle is too slow, it will stop the episode and alert the user of the problem.

**Open questions**

- A cycle should ideally last from originally receiving a state to responding with a new action **in** the plant. If we measure cycle times in the loop, the computations occurring inside the plant (although as minimal as possible) are ignored. So maybe the cycle time determinism should be part of the plant?

## plant

---

## control

## Components

### control

Controllers interact with plants, controlling specific subsystems. Controller can also
be chained, building on top of one another.

### explore

Explorers wrap controllers, providing different exploration strategies,
diverging from a controllers own behavior.

### plant

Plants are either simulators in it self or provide interfaces for interacting
with simulators or real-world systems.

### train

Train models to be used in control.

### utils

Utility functions, including for example specific cost functions, custom gradient descent optimizers and network layers.

## Development

### Setup

```Shell
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tests]" "./psipy-core[tf,automl]"
```

Some jupyter notebooks use the `%matplotlib widget` extension, installation is described here: https://github.com/matplotlib/jupyter-matplotlib#installation

Access to azure storage requires the AZURE_STORAGE_ACCOUNT and
AZURE_STORAGE_ACCESS_KEY environment variables to be set.
Therefore, you can create a .env file and store the two variables there like in the .env.example file.
The environment variables will be then loaded with the pip package python-dotenv in the python file.

### Guidelines

All code is formatted using black and checked using pytest, flake8 and mypy. Running pytest runs all three of those.

```Shell
pytest
```
