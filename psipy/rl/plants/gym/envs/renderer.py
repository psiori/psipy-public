# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Cartpole Gym Renderer

This renderer adds textual information to the rendered cartpole environments that
use the mixin. State, costs, cycle, and action are displayed.
"""

from typing import Any, Callable, Optional

import numpy as np
import pyglet

Viewer: Any = object
try:
    from gymnasium.envs.classic_control import rendering
    from pyglet.gl import glClearColor

    Viewer = rendering.Viewer
except (NameError, pyglet.canvas.xlib.NoSuchDisplayException):
    pass


class CustomViewer(Viewer):
    def __init__(self, width, height, display=None):
        super(CustomViewer, self).__init__(width=width, height=height, display=display)
        self.annotations = {}

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        if hasattr(self, "annotations"):
            for v in self.annotations.values():
                v.draw()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data._current_data, dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen


class RenderMixin:
    """Mixin for Cartpole environments that displays state and cost info on screen."""

    def __init__(self, cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self._cost_func = cost_func
        self.costs = 0
        self.cycle = 0
        self.annotations = [
            "state",
            "cost_label",
            "start",
            "force",
            "step_counter",
            "description",
        ]

    def add_annotations(self, screen_height, screen_width):
        self.viewer.annotations["state"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=20,
            y=screen_height * 2.5 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["cost_label"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=20,
            y=screen_height * 5.0 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["start"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=20,
            y=screen_height * 7.5 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["force"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=screen_width - 25,
            y=screen_height * 2.5 / 40.00,
            anchor_x="right",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["step_counter"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            # x=screen_width - 25,
            # y=screen_height * 5 / 40.00,
            x=screen_width - 25,
            y=screen_height * 7.5 / 40.00,
            anchor_x="right",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["description"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=20,
            y=screen_height * 37.5 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.viewer.annotations["interaction_time"] = pyglet.text.Label(
            "",
            font_size=12,
            # font_name="Courier New",
            x=screen_width - 200,
            y=screen_height - 20,
            anchor_x="left",
            anchor_y="bottom",
            color=(0, 0, 0, 255),
        )

    def print_annotations(self, x, description):
        if "start" in self.annotations:
            self.viewer.annotations[
                "start"
            ].text = f"Start: {[f'{s:+1,.2f}' for s in self.start_state]}"
        if "state" in self.annotations:
            self.viewer.annotations[
                "state"
            ].text = f"State: {[f'{s:+1,.2f}' for s in self.state]}"
        if "cost_label" in self.annotations:
            if self._cost_func is not None:
                self.viewer.annotations[
                    "cost_label"
                ].text = f"Cost: {self._cost_func(x[None, ...])}"
            else:
                self.viewer.annotations["cost_label"].text = "Cost: -not provided-"
        if "force" in self.annotations:
            self.viewer.annotations[
                "force"
            ].text = f"Force: {self.current_force:+1,.2f}"
        if "step_counter" in self.annotations:
            self.viewer.annotations[
                "step_counter"
            ].text = f"Cycle: {self.step_counter:03d}"
        if "description" in self.annotations:
            self.viewer.annotations["description"].text = description or ""
        if "interaction_time" in self.annotations:
            self.viewer.annotations[
                "interaction_time"
            ].text = f"Interaction time: {round(self.cycle * .02,2)}s"

    def render(self, mode="human", description=None):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        self.cycle += 1

        try:
            x_target = self.x_goal * scale + screen_width / 2.0
            x_start = self.x_start * scale + screen_width / 2.0
        except (TypeError, AttributeError):
            x_target = 0
            x_start = 0

        if self.viewer is None:
            self.viewer = CustomViewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

            target = rendering.Line((x_target, 0), (x_target, screen_height))
            target.set_color(0, 0, 0)
            self.viewer.add_geom(target)
            self._target_geom = target

            start = rendering.Line((x_start, 0), (x_start, screen_height))
            start.set_color(255, 0, 0)
            self.viewer.add_geom(start)
            self._start_geom = start

            self.transform = rendering.Transform()

            self.add_annotations(screen_height, screen_width)
        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = np.array(self.state)
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        self._target_geom.start = (x_target, 0)
        self._target_geom.end = (x_target, screen_height)

        self._start_geom.start = (x_start, 0)
        self._start_geom.end = (x_start, screen_height)

        self.print_annotations(x, description)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class RenderMixinTimeOnly(RenderMixin):
    """RenderMixin that only shows the interaction time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotations = ["interaction_time"]
