import gc
import glob
import os
import time
from multiprocessing import Pool
from os.path import join

import tensorflow as tf

from psipy.rl import Loop
from psipy.rl.control import NFQ
from psipy.rl.plant.gym.cartpole_plants import CartPoleUnboundedPlant

# Set CPU as available physical device
tf.config.set_visible_devices([], "GPU")


def eval_arm_models(entry_params):
    experiment_dir, arm, iters, only_last, except_last = entry_params
    print("Creating eval data for individually trained arm models...")
    list_of_models = sorted(
        glob.glob(
            join(experiment_dir, "individual_train", str(arm), "models", "*.zip")
        ),
        key=lambda x: int(x.split("_")[-1].split(".zip")[0]),
    )
    for i, path in enumerate(list_of_models):
        if only_last and i < len(list_of_models) - 1:
            continue
        if except_last and i == len(list_of_models) - 1:
            print(f"Done at {i}")
            break

        print(f"Evaluating model {os.path.split(path)[-1]}")
        nfq = NFQ.load(path)
        nfq.epsilon = 0
        loop = Loop(
            CartPoleUnboundedPlant(swingup=True),
            nfq,
            logdir=join(experiment_dir, "individual_train", str(arm), "eval", str(i)),
        )
        loop.run(episodes=iters, max_episode_steps=200)

        # Collect garbage and clear Keras session due to memory leak in TF 2.0
        gc.collect()
        tf.keras.backend.clear_session()


def get_params(name, arms, iters, only_last, except_last):
    # Check for invalid setup
    if only_last and except_last:
        raise ValueError("Both can not be true!")

    n_arms = len(arms)
    print(f"Evaluating {n_arms} arms ({arms}).")
    entry_params = []
    for arm in range(n_arms):
        entry_params.append((name, arm, iters, only_last, except_last))
    return entry_params


if __name__ == "__main__":
    import argparse, sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment", help="The path to the experiment folder")
        parser.add_argument("iters", help="The number of evaluation iterations")
        parser.add_argument(
            "whicharms", nargs="+", help="Which arms to individually train"
        )
        parser.add_argument(
            "--onlylast",
            action="store_true",
            help="Only evaluate the final model of the given arms",
        )
        parser.add_argument(
            "--exceptlast",
            action="store_true",
            help="Evaluate all models except for the last",
        )
        args = parser.parse_args()
        name = args.experiment
        iters = int(args.iters)
        arms = [int(arm) for arm in args.whicharms]
        only_last = args.onlylast
        except_last = args.exceptlast
    else:
        name = "Sway-Test-2"
        arms = (0, 1, 2, 3, 4, 5, 6, 7)  # example
        iters = 10
        only_last = False
        except_last = False

    n_arms = len(arms)
    entry_params = get_params(name, arms, iters, only_last, except_last)

    s = time.time()
    with Pool(n_arms) as p:
        p.map(eval_arm_models, entry_params)

    print(f"Total time for {n_arms} arms: {round((time.time() - s)/60, 2)}m")
