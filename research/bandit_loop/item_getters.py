import glob
import os
from datetime import datetime
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from psipy.rl.io.batch import Batch, Episode


class Getter:
    """Simple init interface for Getter type classes."""

    def __init__(
        self,
        root_dir: str,
        best_arm: Optional[int],
        n_arms: int,
        has_stability: bool = False,
    ):
        self.root_dir = root_dir
        self.best_arm = best_arm
        self.n_arms = n_arms
        self.has_stability = has_stability


class FinalPolicyGetter(Getter):
    """Get data related to the final policy per bandit arm."""

    def get_final_policy_data(
        self, arm: int, state_channel: Union[str, Tuple[str, ...]] = "pole_cosine"
    ) -> np.ndarray:
        """Gets the data for the final version of each model at the end of training."""
        path = glob.glob(join(self.root_dir, "bandit_data", "evaluation", f"{arm}-*"))[
            0
        ]
        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        ep = Episode.from_hdf5(path, state_channels=state_channel)
        return ep.all_observations

    def get_final_stability_data(self, state_channel: str = "pole_cosine"):
        path = glob.glob(
            join(self.root_dir, "bandit_data", "evaluation", f"*-stability-*")
        )[0]
        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        ep = Episode.from_hdf5(path, state_channels=state_channel)
        return ep.all_observations


class BanditModelGetter(Getter):
    """Get bandit models."""

    def _get_arm_models(self, arm: int):
        """Get paths to all models for the given arm."""
        return sorted(
            glob.glob(join(self.root_dir, f"arm-{arm}", "models", "*.zip")),
            key=lambda x: int(x.split("_")[-1][:-4]),
        )

    def get_all_arm_models(self, arm: int):
        """Get paths to all models for the given arm."""
        return self._get_arm_models(arm)

    def get_last_arm_model(self, arm: int):
        """Get the model name of all last models after training (i.e. model 399)."""
        return self._get_arm_models(arm)[-1]

    def get_last_pulled_arm_model(self, arm: int, chosen_arms: List[int]):
        """Get the model name of the last ran model per arm."""
        # -1 because 0 index
        index = len(chosen_arms) - chosen_arms[::-1].index(arm) - 1
        print(f"Getting last pulled arm model (model {index}) for arm {arm}...")
        return self._get_arm_models(arm)[index]


class ArmEvalGetter(Getter):
    """Get everything related to bandit arm evaluation runs."""

    def _get_eval_data(self, arm: Union[int, str]):
        """Get paths to all folders for the given arm's eval.

        Can also be stability arm.
        """
        return sorted(
            glob.glob(join(self.root_dir, f"arm-{arm}", "eval", "*")),
            key=lambda x: int(os.path.split(x)[-1]),
        )

    def _get_stability_data(self, index):
        return sorted(
            glob.glob(join(self.root_dir, "arm-stability", "eval", "*")),
            key=lambda x: int(os.path.split(x)[-1]),
        )[index]

    def get_final_eval_channel_data(
        self,
        arm: Union[int, str],
        state_channel: Union[str, Tuple] = "pole_cosine",
        n_eps: int = 10,
    ) -> Dict[int, np.ndarray]:
        """Gets final evaluation episode data. ORACLE INFO!"""
        cosines = {}
        if self.has_stability and arm == self.n_arms - 1 or arm == "Stability":
            return self.get_final_stability_eval_channel_data(state_channel, n_eps)
        # Sort the path, although it doesn't matter what order it is in, because it should
        # be deterministic when potentially slicing
        path = sorted(glob.glob(join(self._get_eval_data(arm)[-1], "*.h5")))
        for ep in path[:n_eps]:
            if isinstance(state_channel, str):
                state_channel = (state_channel,)
            episode = Episode.from_hdf5(ep, state_channels=state_channel)
            cosine = episode.all_observations
            cosines[ep] = cosine.squeeze()
        return cosines

    def get_final_best_arm_eval_channel_data(self, state_channel: str = "pole_cosine"):
        return self.get_final_eval_channel_data(self.best_arm, state_channel)

    def get_final_stability_eval_channel_data(
        self, state_channel: str = "pole_cosine", n_eps: int = 10
    ) -> Dict[int, np.ndarray]:
        cosines = {}
        # Sort the path, although it doesn't matter what order it is in, because it should
        # be deterministic when potentially slicing
        path = sorted(glob.glob(join(self._get_stability_data(-1), "*.h5")))
        for ep in path[:n_eps]:
            if isinstance(state_channel, str):
                state_channel = (state_channel,)
            episode = Episode.from_hdf5(ep, state_channels=state_channel)
            cosine = episode.all_observations
            cosines[ep] = cosine.squeeze()
        return cosines

    def get_terminals_and_lengths_final_eval_as_arrays(
        self, arm: Union[int, str]
    ) -> Tuple[List[bool], List[int]]:
        """Returns if eval episode ended in terminal, and how long it was. ORACLE INFO!"""
        terminals = []
        stops = []
        for final in glob.glob(join(self._get_eval_data(arm)[-1], "*.h5")):
            episode = Episode.from_hdf5(final)
            terminals.append(episode.terminals[-1])
            stops.append(len(episode.all_observations))
        return terminals, stops

    def get_all_final_eval_as_batch(
        self, arm: Union[int, str], state_channel: str = "pole_cosine"
    ) -> Batch:
        """Gets all final eval data as a batch. ORACLE INFO!"""

        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        if self.has_stability and self.best_arm == self.n_arms - 1:
            batch = Batch.from_hdf5(
                self._get_stability_data(-1), state_channels=state_channel
            )
        else:
            batch = Batch.from_hdf5(
                self._get_eval_data(arm)[-1], state_channels=state_channel
            )
        return batch.set_minibatch_size(-1).sort()

    def get_all_final_best_eval_as_batch(self, state_channel: str = "pole_cosine"):
        return self.get_all_final_eval_as_batch(self.best_arm, state_channel)

    def _get_current_stability_arm(self, index):
        def sort_by_time(filename):
            date_time = filename.split("-")[-4:-2]
            return datetime.strptime("-".join(date_time), "%y%m%d-%H%M%S")

        ordered_bandit_episodes = sorted(
            glob.glob(join(self.root_dir, "bandit_data", "[!evaluation]*")),
            key=sort_by_time,
        )
        current_episode = ordered_bandit_episodes[index]
        assert "stability" in os.path.split(current_episode)[-1]
        current_time = datetime.strptime(
            "-".join(current_episode.split("-")[-4:-2]), "%y%m%d-%H%M%S"
        )
        current_model = None
        stability_history = sorted(
            os.listdir(join(self.root_dir, "arm-stability", "stability-history")),
            key=lambda x: int(x.split("-")[3]),
        )
        for model in stability_history:
            model_time = datetime.strptime(
                "-".join(model.split("-")[-2:]).split(".zip")[0], "%y%m%d-%H%M%S"
            )
            if model_time <= current_time:
                current_model = model
            elif model_time > current_time:
                break
        assert current_model is not None
        return int(current_model.split("-")[-3])

    def get_all_chosen_eval_as_batches(
        self,
        chosen_arms: List[int],
        state_channels: Tuple[str],
        has_stability: bool = False,
    ):
        batches = []
        for e, arm in enumerate(chosen_arms):
            if has_stability and arm == max(chosen_arms):
                # Stability arm was picked, get the proper eval data
                stab_index = self._get_current_stability_arm(e)
                batches.append(Batch.from_hdf5(self._get_stability_data(stab_index)))
            else:
                batches.append(
                    Batch.from_hdf5(
                        self._get_eval_data(arm)[e], state_channels=state_channels
                    )
                )

        return batches


class BanditDataGetter(Getter):
    """Get bandit sart data."""

    def get_all_bandit_data_as_batch(self, state_channels) -> Batch:
        batch = Batch.from_hdf5(
            join(self.root_dir, "bandit_data"),
            state_channels=state_channels,
            override_mtime=True,
        )
        return batch.set_minibatch_size(-1).sort()


class IndividualTrainGetter(Getter):
    """Get individually trained arm related data."""

    def __init__(self, root_dir, best_arm, n_arms):
        super().__init__(root_dir, best_arm, n_arms, False)
        self._folder_index = self.root_dir.split("-")[-1]

    def _get_individual_data(self, arm: int):
        # Always pulls the individual arms from the SwingUp-Base data
        return sorted(
            glob.glob(join(self.root_dir, "individual_train", str(arm), "eval", "*",)),
            key=lambda x: int(os.path.split(x)[-1]),
        )

    def get_all_sart_data_as_batch(self, arm, state_channels=("pole_cosine",)) -> Batch:
        batch = Batch.from_hdf5(
            join(self.root_dir, "individual_train", str(arm), "sart"),
            state_channels=state_channels,
            override_mtime=True,
        )
        return batch.set_minibatch_size(-1).sort()

    def get_best_arm_final_eval_as_batch(self, state_channel: str = "pole_cosine"):
        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        batch = Batch.from_hdf5(
            self._get_individual_data(self.best_arm)[-1], state_channels=state_channel,
        )
        return batch.set_minibatch_size(-1).sort()

    def get_best_arm_final_eval_data(self, state_channel: str = "pole_cosine"):
        """Gets final retrained evaluation episode data. ORACLE INFO!"""
        cosines = {}
        for final in glob.glob(
            join(self._get_individual_data(self.best_arm)[-1], "*.h5")
        ):
            episode = Episode.from_hdf5(final, state_channels=(state_channel,))
            cosine = episode.all_observations
            cosines[final] = cosine.squeeze()
        return cosines

    def get_final_eval_data(self, arm: int, state_channel: str = "pole_cosine"):
        """Gets final evaluation episode data. ORACLE INFO!"""
        cosines = {}
        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        for ep in glob.glob(join(self._get_individual_data(arm)[-1], "*.h5")):
            episode = Episode.from_hdf5(ep, state_channels=state_channel)
            cosine = episode.all_observations
            cosines[ep] = cosine.squeeze()
        return cosines

    def get_terminals_and_lengths_final_eval_as_arrays(
        self, arm: Union[int, str]
    ) -> Tuple[List[bool], List[int]]:
        """Returns if eval episode ended in terminal, and how long it was. ORACLE INFO!"""
        terminals = []
        stops = []
        for final in glob.glob(join(self._get_individual_data(arm)[-1], "*.h5")):
            episode = Episode.from_hdf5(final)
            terminals.append(episode.terminals[-1])
            stops.append(len(episode.all_observations))
        return terminals, stops

    def get_all_final_eval_as_batch(
        self, arm: Union[int, str], state_channel: str = "pole_cosine"
    ) -> Batch:
        """Gets all final eval data as a batch. ORACLE INFO!"""
        if isinstance(state_channel, str):
            state_channel = (state_channel,)
        batch = Batch.from_hdf5(
            self._get_individual_data(arm)[-1], state_channels=state_channel
        )
        return batch.set_minibatch_size(-1).sort()

    def get_all_eval_as_batches(self, arm: Union[int, str], state_channels: Tuple[str]):
        batches = []
        for episode in self._get_individual_data(arm):
            batches.append(Batch.from_hdf5(episode, state_channels=state_channels))

        return batches
