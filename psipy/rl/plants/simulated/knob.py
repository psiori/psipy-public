import math
import random
from typing import Callable, Optional, Tuple, Type
import numpy as np
from psipy.rl.core.plant import Action, Plant, State
from psipy.rl.io.batch import Episode

# This plant is for testing implementaitons of controllers easily on a 
# 1-dimensional continous state space with 1-dimensional actions.
# It offers a simple knob that needs to be turned to a zero position from
# random station points. It can be turned in both directions by 0-1 degrees
# per step. It has a variant with continuous actions and one with discrete
# actions. It offers two default cost functions, one where the imediate cost
# is proportional to the angular distance of the present position to the target
# angle. This cost function is meant as an initial sanity check for new learning
# algorithm implementaitons, as optimizing immediate rewards is good enough for
# finding a solution. The other cost function offers delayed reward, where the
# cost is the constant default step cost outside [-1, 0] degrees and 
# propotional to the distance to the target angle inside [-1, 0] degrees.


class KnobAction(Action):
    """Parent class of knob actions."""
    pass


class DiscreteKnobAction(KnobAction):
    """Action with left actions right actions"""
    dtype = "discrete"
    channels = ("turn",)
    legal_values = ((-2.0, -0.01, 0.0, 0.01, 2.0),)


class ContinuousKnobAction(KnobAction):
    """Action with continuous values"""
    dtype = "continuous"
    channels = ("turn",)
    legal_values = ((-2.0, 2.0),)


class KnobState(State):
    """Knob state encoding the position of the knob in [-180, 180] degrees."""
    _channels = (
        "position",
        "turn_ACT",
    )


def make_default_cost_function(default_step_cost: float = 0.01,
                               delayed_reward: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    def cost_function(state: np.ndarray) -> np.ndarray:
        position = state["position"]

        cost = default_step_cost * (abs(position) / 180.0)

        if delayed_reward:
            if abs(position) > 1.0:
                cost = default_step_cost

        return cost

    return cost_function



class Knob(Plant[KnobState, KnobAction]):
    """Knob plant.
    """

    def __init__(
        self,
        cost_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        state_type: Type[KnobState] = KnobState,
        action_type: Type[KnobAction] = ContinuousKnobAction,
        render_mode: str = "human",
        do_not_reset: bool = False,
    ):
        if cost_function is None:
            cost_function = make_default_cost_function()
            print("Knob is using default cost function")
      
        super().__init__(cost_function=cost_function)

        self.renderable = True
        self.render_mode = render_mode

        self.do_not_reset = do_not_reset

        self.state_type = state_type
        self.action_type = action_type

        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400

        self.reset()


    def _get_next_state(self, state: KnobState, action: KnobAction) -> KnobState:

        position, turn_ACT = self._current_state.as_array()

        action = action["turn"]
        self.current_action = action

        position = ((position + 180.0 + action) % 360.0) - 180.0

        return KnobState([position, turn_ACT], 0.0, False) # there are no terminal states
    

    def notify_episode_stops(self) -> bool:
        if not self.do_not_reset:
            self.reset()
        return True
    
    def reset(self):
        position = random.random() * 360.0 - 180.0
        self.current_action = 0.0

        state = [position, 0.0]
        self._current_state = KnobState(state, 0.0, False)
        self._current_state.cost = self._cost_function(self._current_state)

        return self._current_state
    
    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed'
            ) from e

        if self.screen is None:
            pygame.init()
            if True: # self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self._current_state is None:
            return None

        # Get current knob position
        position = self._current_state["position"]

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Knob parameters
        knob_center_x = self.screen_width // 2
        knob_center_y = self.screen_height // 2
        knob_radius = 100
        outer_ring_radius = knob_radius + 20
        inner_ring_radius = knob_radius - 10

        # Draw outer black ring (surrounding marking)
        gfxdraw.aacircle(
            self.surf,
            knob_center_x,
            knob_center_y,
            outer_ring_radius,
            (0, 0, 0)
        )
        gfxdraw.filled_circle(
            self.surf,
            knob_center_x,
            knob_center_y,
            outer_ring_radius,
            (0, 0, 0)
        )

        # Draw inner white circle (knob face)
        gfxdraw.aacircle(
            self.surf,
            knob_center_x,
            knob_center_y,
            knob_radius,
            (255, 255, 255)
        )
        gfxdraw.filled_circle(
            self.surf,
            knob_center_x,
            knob_center_y,
            knob_radius,
            (255, 255, 255)
        )

        # Draw inner black ring
        gfxdraw.aacircle(
            self.surf,
            knob_center_x,
            knob_center_y,
            inner_ring_radius,
            (0, 0, 0)
        )
        gfxdraw.filled_circle(
            self.surf,
            knob_center_x,
            knob_center_y,
            inner_ring_radius,
            (0, 0, 0)
        )

        # Draw center white circle
        center_radius = inner_ring_radius - 10
        gfxdraw.aacircle(
            self.surf,
            knob_center_x,
            knob_center_y,
            center_radius,
            (255, 255, 255)
        )
        gfxdraw.filled_circle(
            self.surf,
            knob_center_x,
            knob_center_y,
            center_radius,
            (255, 255, 255)
        )

        # Draw position indicator (red triangle pointing to current position)
        indicator_length = knob_radius - 15
        indicator_angle = math.radians(position)
        
        # Calculate indicator tip position
        tip_x = knob_center_x + indicator_length * math.sin(indicator_angle)
        tip_y = knob_center_y - indicator_length * math.cos(indicator_angle)
        
        # Calculate indicator base positions (small triangle)
        base_angle1 = indicator_angle + math.radians(15)
        base_angle2 = indicator_angle - math.radians(15)
        base_length = 15
        
        base1_x = knob_center_x + base_length * math.sin(base_angle1)
        base1_y = knob_center_y - base_length * math.cos(base_angle1)
        base2_x = knob_center_x + base_length * math.sin(base_angle2)
        base2_y = knob_center_y - base_length * math.cos(base_angle2)
        
        # Draw red indicator triangle
        indicator_coords = [(tip_x, tip_y), (base1_x, base1_y), (base2_x, base2_y)]
        gfxdraw.aapolygon(self.surf, indicator_coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, indicator_coords, (255, 0, 0))

        # Draw tick marks and labels
        font = pygame.font.Font(None, 24)
        
        # Draw major tick marks every 45 degrees
        for angle in range(-180, 181, 45):
            rad_angle = math.radians(angle)
            outer_x = knob_center_x + (knob_radius + 5) * math.sin(rad_angle)
            outer_y = knob_center_y - (knob_radius + 5) * math.cos(rad_angle)
            inner_x = knob_center_x + (knob_radius - 5) * math.sin(rad_angle)
            inner_y = knob_center_y - (knob_radius - 5) * math.cos(rad_angle)
            
            gfxdraw.line(self.surf, int(outer_x), int(outer_y), int(inner_x), int(inner_y), (0, 0, 0))

        # Draw labels
        # 0 at top (target position)
        label_0 = font.render("0", True, (0, 0, 0))
        label_0_rect = label_0.get_rect(center=(knob_center_x, knob_center_y - knob_radius - 25))
        self.surf.blit(label_0, label_0_rect)
        
        # -180 at bottom left
        label_neg180 = font.render("-180", True, (0, 0, 0))
        label_neg180_rect = label_neg180.get_rect(center=(knob_center_x - 35, knob_center_y + knob_radius + 35))
        self.surf.blit(label_neg180, label_neg180_rect)
        
        # 180 at bottom right
        label_180 = font.render("180", True, (0, 0, 0))
        label_180_rect = label_180.get_rect(center=(knob_center_x + 35, knob_center_y + knob_radius + 35))
        self.surf.blit(label_180, label_180_rect)

        # Draw current position text
        position_text = font.render(f"Position: {position:.1f}°", True, (0, 0, 0))
        position_rect = position_text.get_rect(center=(knob_center_x, knob_center_y + knob_radius + 55))
        self.surf.blit(position_text, position_rect)

        # Draw current cost text
        if hasattr(self, '_current_state') and hasattr(self._current_state, 'cost'):
            cost_text = font.render(f"Cost: {self._current_state.cost:.5f}", True, (200, 0, 0))
            cost_rect = cost_text.get_rect(center=(knob_center_x, knob_center_y + knob_radius + 70))
            self.surf.blit(cost_text, cost_rect)

        # Draw action indicator if there's a current action
        if hasattr(self, 'current_action') and self.current_action != 0:
            action_text = font.render(f"Action: {self.current_action:.3f}", True, (0, 100, 200))
            action_rect = action_text.get_rect(center=(knob_center_x, knob_center_y + knob_radius + 85))
            self.surf.blit(action_text, action_rect)

        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(100)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

