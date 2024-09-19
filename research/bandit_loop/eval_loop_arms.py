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
    print("Creating eval data for arm models during training...")
    list_of_models = sorted(
        glob.glob(join(experiment_dir, f"arm-{str(arm)}", "models", "*.zip")),
        key=lambda x: int(x.split("_")[-1].split(".zip")[0]),
    )
    # figure, ax = plt.subplots(figsize=(5, 3))
    # mem = []
    for i, path in enumerate(list_of_models):
        if only_last and i < len(list_of_models) - 1:  # < 399
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
            logdir=join(experiment_dir, f"arm-{str(arm)}", "eval", str(i)),
        )
        loop.run(episodes=iters, max_episode_steps=200)
        # loop.run(episodes=1, max_episode_steps=10)
        # reset_session()
        # Collect garbage and clear Keras session due to memory leak in TF 2.0
        gc.collect()
        tf.keras.backend.clear_session()
        # curr_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
        # print(curr_mem, f"GB used in process {os.getpid()}")
        # ax.clear()
        # mem.append(curr_mem)
        # ax.plot(mem)
        # plt.pause(.01)
        # time.sleep(1) # ensure different sart names


def eval_stability_arm(experiment_dir, iters):
    try:
        # Evaluate the stability arm
        stability_arm_model = sorted(
            glob.glob(
                join(experiment_dir, "arm-stability", "stability-history", "*.zip")
            ),
            key=lambda x: int(os.path.split(x)[-1].split("-")[-3]),
        )
        print(f"Evaluating {len(stability_arm_model)} stability arms...")
        for i, stability_arm in enumerate(stability_arm_model):
            nfq = NFQ.load(join(stability_arm))
            nfq.epsilon = 0
            loop = Loop(
                CartPoleUnboundedPlant(swingup=True),
                nfq,
                logdir=join(experiment_dir, f"arm-stability", "eval", str(i)),
            )
            loop.run(episodes=iters, max_episode_steps=200)
        print("Evaluated the stability arms.")
    except Exception as e:
        print(f"No stability arm found ({e})")


def get_params(name, n_arms, iters, only_last, except_last):
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
        parser.add_argument(
            "--stability",
            action="store_true",
            help="Only evaluate the stability models",
        )
        args = parser.parse_args()
        name = args.experiment
        iters = int(args.iters)
        arms = [arm for arm in args.whicharms]
        only_last = args.onlylast
        only_stability = args.stability
        except_last = args.exceptlast
    else:
        name = "Sway-Test-2"
        arms = (0, 1, 2, 3, 4, 5, 6, 7)  # example
        iters = 10
        only_last = False
        except_last = False
        only_stability = False

    # Check for invalid setup
    if only_last and except_last:
        raise ValueError("Both can not be true!")

    if not only_stability:
        n_arms = len(arms)
        arms = [int(arm) for arm in arms]
        print(f"Evaluating {n_arms} arms ({arms}).")
        entry_params = get_params(name, n_arms, iters, only_last, except_last)

        s = time.time()
        with Pool(n_arms) as p:
            p.map(eval_arm_models, entry_params)
        print(f"Total time for {n_arms} arms: {round((time.time() - s) / 60, 2)}m")

    eval_stability_arm(name, iters)
