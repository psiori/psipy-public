# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""PSIORI Reinforcement Learning Toolbox
=====================================

Toolbox for closed loop control using reinforcement learning, specifically
using deep neural network based function approximators trained using batch
reinforcement learning approaches.

The toplevel interface consists of three primary concepts: The loop,
plants and controllers. Additionally, different helper methods for i/o, data
preprocessing and visualization are provided.

The central focus in many design decisions is to never block the loop or even
risk it stopping. Therefore, all i/o like writing to disk or receiving data
from external essential processes runs in its own threads. Some supervision
of the loop is provided through a terminal user interface, which communicates
with the loop in non-blocking fashion through :class:`zmq.Socket` instances.

In contrast to many reinforcement learning libraries, in psipy, system
interaction is completely independent from agent training. The focus of the
learning component of this library lies in Batch or Growing Batch Reinforcement
Learning, which has proven successful in data efficient interaction with
real-world control tasks.

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

Read more:  :mod:`psipy.rl.plant`

Control
-------

Read more: :mod:`psipy.rl.control`

"""


from psipy.rl import control, io, plant
from psipy.rl.cycle_manager import CM
from psipy.rl.loop import Loop

__all__ = [
    "CM",
    "control",
    "exceptions",
    "io",
    "loop",
    "Loop",
    "plant",
    "scripts",
    "visualization",
]
