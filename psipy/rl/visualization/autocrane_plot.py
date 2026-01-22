import numpy as np
from matplotlib import pyplot as plt

from psipy.core.notebook_tools import is_notebook
from psipy.rl.io.batch import Episode


class AutocraneTrolleyTrajectoryPlot:
    def __init__(
        self,
        filename=None,
        trolley_position_idx: int = 0,
        trolley_vel_idx: int = 1,
        trolley_sway_idx: int = 2,
        trolley_sway_vel_idx: int = 3,
        hoist_position_idx: int = 4,
        hoist_vel_idx: int = 5,
        do_plot_hoist: bool = False,
        do_display: bool = True,
        trolley_margin: float = 0.3,
        hoist_margin: float = 0.1,
        sway_margin: float = 0.04,
        trolley_set_point_delta_idx: int = 21,
        distance_cost_func=None,
        sway_cost_func=None,
        position_zero_threshold: float | None = None,
        sway_zero_threshold: float | None = None,
    ):
        self.trolley_set_point_delta_idx = trolley_set_point_delta_idx
        self.trolley_position_idx = trolley_position_idx
        self.trolley_vel_idx = trolley_vel_idx
        self.trolley_sway_idx = trolley_sway_idx
        self.trolley_sway_vel_idx = trolley_sway_vel_idx
        self.hoist_position_idx = hoist_position_idx
        self.hoist_vel_idx = hoist_vel_idx

        self.trolley_margin = trolley_margin
        self.hoist_margin = hoist_margin
        self.sway_margin = sway_margin

        self.do_plot_hoist = do_plot_hoist
        self.episode = None
        self.filename = filename
        self.fig = None
        self.axs = None
        self.dirty = True
        self.do_display = do_display
        self.episode_num = None
        self.title_string = None
        self.distance_cost_func = distance_cost_func
        self.sway_cost_func = sway_cost_func
        self.position_zero_threshold = position_zero_threshold
        self.sway_zero_threshold = sway_zero_threshold

    def update(
        self,
        episode: Episode,
        episode_num: int | None = None,
        title_string: str | None = None,
        is_random: np.ndarray | None = None,
    ):
        self.episode = episode
        self.episode_num = episode_num
        self.title_string = title_string
        self.is_random = is_random
        self.dirty = True

    def plot(self):
        self._maybe_plot()

    def save(self, filename=None):
        filename = self.filename if filename is None else filename
        self._maybe_plot()
        if self.fig is not None and filename is not None:
            self.fig.savefig(filename)

    def _is_notebook(self):
        return (
            is_notebook()
        )  # there is a shared implementation provided by psipy.core.notebook_tools

    def _maybe_plot(self):
        if not self.dirty or self.episode is None or self.episode.observations is None:
            return

        if self.fig is None:
            if self.do_plot_hoist:
                self.fig, self.axs = plt.subplots(7, figsize=(10, 12))
            else:
                # 6 subplots: position error, sway, control, position cost, sway cost, total cost
                self.fig, self.axs = plt.subplots(6, figsize=(10, 10))
            self.fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        elif not self._is_notebook():
            plt.figure(self.fig.number)

        if self.axs is None:
            return

        for ax in self.axs:
            ax.clear()

        x = self.episode.observations[:, self.trolley_position_idx]
        x_s = self.episode.observations[:, self.trolley_vel_idx]
        # t = self.episode.observations[:, self.pole_theta_idx]
        sway = self.episode.observations[:, self.trolley_sway_idx]
        a = self.episode._actions[:, 0]
        cost = self.episode.costs

        if self.do_plot_hoist:
            hoist_pos = self.episode.observations[:, self.hoist_position_idx]
            hoist_vel = self.episode.observations[:, self.hoist_vel_idx]
            hoist_action = self.episode._actions[:, 1]

        axis_counter = 0

        set_point_delta = self.episode.observations[:, self.trolley_set_point_delta_idx]
        position_threshold = (
            self.position_zero_threshold
            if self.position_zero_threshold is not None
            else self.trolley_margin
        )
        self.axs[axis_counter].plot(set_point_delta, label="position_error")
        self.axs[axis_counter].axhline(0, color="grey", linestyle=":", label="target")
        # Add shaded area around target to show acceptable deviation using actual threshold
        self.axs[axis_counter].axhspan(
            -position_threshold,
            position_threshold,
            color="grey",
            alpha=0.2,
            label="zero cost range",
        )
        # Add darker lines at boundaries
        self.axs[axis_counter].axhline(-position_threshold, color="grey", alpha=0.5)
        self.axs[axis_counter].axhline(position_threshold, color="grey", alpha=0.5)
        self.axs[axis_counter].set_title("Position Error")
        self.axs[axis_counter].set_ylabel("Position Error (m)")
        self.axs[axis_counter].set_ylim((-4.5, 4.5))
        self.axs[axis_counter].legend()
        axis_counter += 1

        if self.do_plot_hoist:
            self.axs[axis_counter].plot(hoist_pos, label="hoist_position")
            self.axs[axis_counter].axhline(
                0, color="grey", linestyle=":", label="target"
            )
            # Add blue shaded area of ±0.4m around target to show acceptable deviation
            self.axs[axis_counter].axhspan(
                -self.hoist_margin,
                self.hoist_margin,
                color="grey",
                alpha=0.2,
                label="acceptable range",
            )
            # Add darker blue lines at boundaries
            self.axs[axis_counter].axhline(-self.hoist_margin, color="grey", alpha=0.5)
            self.axs[axis_counter].axhline(self.hoist_margin, color="grey", alpha=0.5)
            self.axs[axis_counter].set_ylim((-1.0, 1.0))
            self.axs[axis_counter].set_title("hoist_position")
            self.axs[axis_counter].set_ylabel("Position")
            self.axs[axis_counter].legend()
            axis_counter += 1

        sway_threshold = (
            self.sway_zero_threshold
            if self.sway_zero_threshold is not None
            else self.sway_margin
        )
        self.axs[axis_counter].plot(sway, label="sway")
        self.axs[axis_counter].axhline(0, color="grey", linestyle=":", label="target")
        # Add shaded area around target to show acceptable deviation using actual threshold
        self.axs[axis_counter].axhspan(
            -sway_threshold,
            sway_threshold,
            color="grey",
            alpha=0.2,
            label="zero cost range",
        )
        # Add darker lines at boundaries
        self.axs[axis_counter].axhline(-sway_threshold, color="grey", alpha=0.5)
        self.axs[axis_counter].axhline(sway_threshold, color="grey", alpha=0.5)
        self.axs[axis_counter].set_title("Sway")
        self.axs[axis_counter].set_ylabel("Sway (cosine)")
        self.axs[axis_counter].set_ylim((-0.15, 0.15))
        self.axs[axis_counter].legend()
        axis_counter += 1

        self.axs[axis_counter].plot(a, label="trolley action")
        self.axs[axis_counter].plot(
            x_s, color="black", alpha=0.4, label="trolley_velocity"
        )
        if self.is_random is not None and len(self.is_random) > 0:
            is_random_flat = (
                self.is_random.ravel()
                if len(self.is_random.shape) > 1
                else self.is_random
            )
            # Align is_random with actions array (accounting for lookback)
            # Actions array excludes initial lookback-1 observations in episode.observations
            # but _actions includes all actions
            min_len = min(len(is_random_flat), len(a))
            is_random_aligned = is_random_flat[:min_len]
            a_aligned = a[:min_len]
            random_indices = np.where(is_random_aligned)[0]
            if len(random_indices) > 0:
                self.axs[axis_counter].scatter(
                    random_indices,
                    a_aligned[random_indices],
                    color="red",
                    marker="x",
                    s=50,
                    zorder=5,
                    label="random action",
                )
        self.axs[axis_counter].axhline(0, color="grey", linestyle=":")
        self.axs[axis_counter].set_title("Control")
        self.axs[axis_counter].set_ylabel("Velocity")
        self.axs[axis_counter].set_ylim((-0.5, 0.5))
        self.axs[axis_counter].legend(loc="upper left")
        axis_counter += 1

        if self.do_plot_hoist:
            self.axs[axis_counter].plot(hoist_action, label="hoist action")
            self.axs[axis_counter].plot(
                hoist_vel, color="black", alpha=0.4, label="hoist_velocity"
            )
            self.axs[axis_counter].axhline(0, color="grey", linestyle=":")
            self.axs[axis_counter].set_title("Hoist Control")
            self.axs[axis_counter].set_ylabel("Velocity")
            self.axs[axis_counter].set_ylim((-0.2, 0.2))
            self.axs[axis_counter].legend(loc="upper left")
            axis_counter += 1

        # Calculate individual costs if cost functions are available
        position_costs = None
        sway_costs = None
        if self.distance_cost_func is not None and self.sway_cost_func is not None:
            # Extract relevant channels from episode observations
            # Plant-internal channel indices: trolley_set_point_delta=21, grapple_sway_trolley=9,
            # trolley_limit_dist_left=16, trolley_limit_dist_right=17
            # Controller state channels: [trolley_set_point_delta, trolley_vel, grapple_sway_trolley,
            # trolley_limit_dist_left, trolley_limit_dist_right, trolley_target_vel_ACT]
            n_steps = len(self.episode.observations)
            states = np.zeros((n_steps, 6))
            states[:, 0] = self.episode.observations[
                :, self.trolley_set_point_delta_idx
            ]  # position_idx
            states[:, 1] = self.episode.observations[
                :, self.trolley_vel_idx
            ]  # trolley_vel (not used in cost)
            states[:, 2] = self.episode.observations[
                :, self.trolley_sway_idx
            ]  # cosine_idx
            # Find limit distance indices in plant-internal channels
            # trolley_limit_dist_left is at index 16, trolley_limit_dist_right is at index 17
            states[:, 3] = self.episode.observations[:, 16]  # left_distance_idx
            states[:, 4] = self.episode.observations[:, 17]  # right_distance_idx
            states[:, 5] = self.episode.observations[
                :, 25
            ]  # trolley_target_vel_ACT (not used in cost)

            position_costs = self.distance_cost_func(states)
            sway_costs = self.sway_cost_func(states)

        if position_costs is not None:
            self.axs[axis_counter].plot(
                position_costs, label="position_cost", color="blue"
            )
            self.axs[axis_counter].set_title("Position Cost")
            self.axs[axis_counter].set_ylabel("Cost")
            self.axs[axis_counter].set_ylim((0.0, 0.02))
            self.axs[axis_counter].legend()
            axis_counter += 1

        if sway_costs is not None:
            self.axs[axis_counter].plot(sway_costs, label="sway_cost", color="red")
            self.axs[axis_counter].set_title("Sway Cost")
            self.axs[axis_counter].set_ylabel("Cost")
            self.axs[axis_counter].set_ylim((0.0, 0.02))
            self.axs[axis_counter].legend()
            axis_counter += 1

        if cost is not None:
            self.axs[axis_counter].plot(cost, label="total_cost", color="black")
            self.axs[axis_counter].set_title("Total Cost")
            self.axs[axis_counter].set_ylabel("Cost")
            self.axs[axis_counter].set_ylim((0.0, 0.03))
            self.axs[axis_counter].legend()
            axis_counter += 1

        if self.episode_num is None:
            title = "Trolley Control"
        else:
            title = "Trolley Control, Episode {}".format(self.episode_num)

        if self.title_string:
            title = title + " - " + self.title_string

        self.fig.suptitle(title)

        if self.do_display:
            if self._is_notebook():
                self.fig.canvas.draw()
            else:
                # This is what makes it live
                # If you get
                #   AttributeError: type object 'FigureCanvasBase'
                #     has no attribute 'start_event_loop_default'
                # you are in a notebook and either you did not set
                # 'in_notebook' to True, or it wasn't detected correctly.
                plt.pause(0.01)

        self.dirty = False
