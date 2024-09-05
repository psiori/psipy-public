import multiprocessing as mp
import os
import time
from multiprocessing import Pool


def train_bandit_loops(entry_params):
    experiment_i, main_dir = entry_params
    if "Stability" in main_dir:
        arg = "--stability"
    elif "Bad" in main_dir:
        arg = "--badarm"
    elif "Random" in main_dir:
        arg = "--random"
    elif "Adversarial" in main_dir:
        arg = "--adversarial"
    else:
        arg = ""
    name = f"{main_dir}-{experiment_i}"
    print(name)
    # train_bandit_loop(name, bad, stability, random, plot)
    cmd1 = f"python3 bandit_loop_experiment.py {name} {arg}"
    cmd2 = f"python3 eval_loop_arms.py {name} 10 0 1 2 3 4"
    cmd3 = f"python3 bandit_train_all_arms_experiment.py {name} 0 1 2 3 4"
    cmd4 = f"python3 eval_individual_arms.py {name} 10 0 1 2 3 4"
    zipcmd = f"zip -r {name}.zip {name}"  # " && mv {name}.zip output"
    if arg == "":
        cmd = f"{cmd1} && {cmd2} && {cmd3} && {cmd4} && {zipcmd}"
    else:
        cmd = f"{cmd1} && {cmd2} && {zipcmd}"
    # cmd = f"{cmd3} && {cmd4}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    import argparse, sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "experiment", help="The main directory name to store results"
        )
        parser.add_argument("which", nargs="+", help="Which seeds to run")
        args = parser.parse_args()
        main_dir = args.experiment
        which = [int(i) for i in args.which]
    else:
        main_dir = "SwingUpMP-TEST"
        which = [1, 2, 3, 4, 10]

    entry_params = []
    for experiment in which:
        entry_params.append((experiment, main_dir))

    mp.set_start_method("spawn")
    s = time.time()
    with Pool(len(entry_params)) as p:
        p.map(train_bandit_loops, entry_params)
    print(f"Total time: {round((time.time()-s)/60,2)}m")
