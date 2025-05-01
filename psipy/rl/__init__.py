"""PSIORI Reinforcement Learning Tools
======================================

Tools for closed loop control using reinforcement learning, specifically
using deep neural network based function approximators trained using batch
reinforcement learning approaches.

The toplevel interface consists of three primary concepts: The loop,
plants and controllers. Additionally, different helper methods for i/o,
data preprocessing and visualization are provided.

The central focus in many design decisions is to never block the loop or even
risk it stopping. Therefore, all i/o like writing to disk or receiving data
from external essential processes runs in its own threads. Some supervision
of the loop is provided through a terminal user interface, which communicates
with the loop in non-blocking fashion through :class:`zmq.Socket` instances.

In contrast to many reinforcement learning libraries, in psipy, system
interaction is completely independent from agent training. The focus of
the learning component of this library lies in Batch or Growing Batch
Reinforcement Learning, which has proven successful in data efficient
interaction with real-world control tasks.

Since we're often working in industrial environments, controlling motors,
pumps, valves and other industrial equipment, we adapted some of the terminology
of that domain. We refer to plants (not environments) and controllers
(not agents). We make a point to use the terms "state" (and "observations")
but we adapted the term "channel" to refer to the individual input
dimensions of a state, as this is more common terminology when working
with Programmable Logic Controllers (PLC) and Supervisory Control and Data
Acquisition (SCADA) systems.

We also make a point of collecting all transition data (SARS's) together
with their semantig information, thus using named channels always. This
allows us to easily collect additional data information per cycle and to
select, change or "project" the actual state information later.

Loop
-----

The Loop is the main connection between the plant and the controller. It
operates in episodes, starting and stopping all components cleanly
between episodes and on process shutdown. Each episode is a single run from
system startup (or, if it is a continuously running system, loop startup) to
system shutdown, which is commonly described as a terminal state.

Read more: :mod:`psipy.rl.loop`

Plant
-----

Read more:  :mod:`psipy.rl.core.plant`

Control
-------

Read more: :mod:`psipy.rl.core.control`

"""


# Imports
from psipy.rl import io, core, visualization   
#from psipy.rl.core.cycle_manager import CM
from psipy.rl.loop import Loop

# Define what should be available when importing from psipy.rl
__all__ = [
    "core",
    "io",
    "loop",
    "Loop",
    "visualization",
#    "CM",
]
