# PSIORI PACT
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Evaluate a Cartpole Hardware controller for infinite steps."""
import logging
import os
import sys

import numpy as np

from psipy.rl import Loop
from psipy.rl.control import NFQs
from psipy.rl.control.nfq import NFQ, tanh2
from psipy.rl.loop import LoopPrettyPrinter

from cartpole_control.plant.cartpole_plant import SwingupPlant

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ModelTypes = [NFQ, NFQs]


def costfunc(states: np.ndarray) -> np.ndarray:
    theta = states[:, 2].copy()
    costs = tanh2(theta, C=1 / 200, mu=0.1)

    return costs


def evaluate_controller(hostname: str, model_path: str, model_type: ModelTypes) -> None:
    """Evaluate a controller to rate its policy.

    Args:
        hostname: hostname for zmq connection to BUSY
        model_path: path to the saved model .zip
        model_type: base model class of the desired model
    """
    pp = LoopPrettyPrinter(costfunc)
    model = model_type.load(model_path)
    model.epsilon = 0
    # Make sure the speeds matches the speed the model was
    # trained on! (if not continuous)
    loop = Loop(
        SwingupPlant(
            hostname,
            5555,
            5556,
            speed_values=[200],
            angle_terminals=False,
            continuous=True,
            backward_compatible=True,
            cost_func=costfunc,
            controller=model,
        ),
        model,
        "Evaluation",
        "Evaluation-SART",
    )

    while True:
        stop = loop.run_episode(1, pretty_printer=pp)
        if stop:
            break


if __name__ == "__main__":
    model_path = "../models/balance/" \
                 "200501-000000-balance-NFQ" \
                 "-v20-v100-v200-v400-v600-v800-v1000-" \
                 "t20-reallysteadycenterrename.zip"
    evaluate_controller("pact-cube", model_path, NFQ)
