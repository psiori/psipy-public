from typing import Callable, Optional
import numpy as np
from psipy.rl.core.plant import Action, Plant, State

# A wumpus-world like environment on a two d grid with 2d actions; that is, the agent
# can control horizontal and vertical movement indepently from each other.
#
# Present implementation is meant as a test and offers only free fields and a single,
# fixed goal field.


class WumpusState(State):
    _channels = (
        "x",
        "y",
    )

class WumpusAction(Action):
    dtype = "discrete"
    channels = (
        "move_x",
        "move_y",
    )
    legal_values = (
        (-1, 0, 1),  # move_x
        (-1, 0, 1),  # move_y
    )



class WumpusPlant(Plant[WumpusState, WumpusAction]):
    def __init__(self, size: int = 4, 
                 cost_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 render_mode: str = "human"):
        self.action_type = WumpusAction
        self.state_type = WumpusState

        self.size = size
        self.gold_position = (size-1, size-1)

        self.render_mode = render_mode
        self.renderable = True

        if cost_function is None:
            cost_function = self.make_default_cost_function(self)
        
        super().__init__(cost_function=cost_function)

        # Render-related attributes
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 600

        self.reset()

    @classmethod
    def make_default_cost_function(cls, world: "WumpusPlant") -> Callable[[np.ndarray], np.ndarray]:
        def cost_function(state: np.ndarray) -> np.ndarray:
            x, y = state.as_array()
            if x == world.gold_position[0] and y == world.gold_position[1]:
                return 0.0
            else:
                return 0.01

        return cost_function

    def _get_next_state(self, state: WumpusState, action: WumpusAction) -> WumpusState:
        x, y = state.as_array()
        move_x, move_y = action.as_array()

        x = min(max(x + move_x, 0), self.size - 1)
        y = min(max(y + move_y, 0), self.size - 1)

        terminal = x == self.gold_position[0] and y == self.gold_position[1]

        return WumpusState([x, y], terminal=terminal)
    
    def notify_episode_stops(self) -> bool:
        self.reset()
        return True

    def reset(self):
        self._current_state = WumpusState([0, 0], terminal=False)
        self._current_state.cost = self._cost_function(self._current_state)

        return self._current_state

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError(
                'pygame is not installed'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Calculate cell size and margins
        margin = 50
        grid_size = min(self.screen_width, self.screen_height) - 2 * margin
        cell_size = grid_size // self.size
        
        # Colors
        BACKGROUND_COLOR = (255, 255, 255)  # White
        GRID_COLOR = (200, 200, 200)        # Light gray
        WUMPUS_COLOR = (139, 69, 19)        # Brown
        GOLD_COLOR = (255, 215, 0)          # Gold
        TEXT_COLOR = (0, 0, 0)              # Black

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill(BACKGROUND_COLOR)

        # Draw grid
        for i in range(self.size + 1):
            # Vertical lines
            x = margin + i * cell_size
            pygame.draw.line(self.surf, GRID_COLOR, (x, margin), (x, margin + grid_size))
            # Horizontal lines
            y = margin + i * cell_size
            pygame.draw.line(self.surf, GRID_COLOR, (margin, y), (margin + grid_size, y))

        # Draw gold position
        gold_x = margin + self.gold_position[0] * cell_size + cell_size // 2
        gold_y = margin + self.gold_position[1] * cell_size + cell_size // 2
        gold_radius = cell_size // 3
        pygame.draw.circle(self.surf, GOLD_COLOR, (gold_x, gold_y), gold_radius)
        pygame.draw.circle(self.surf, TEXT_COLOR, (gold_x, gold_y), gold_radius, 2)

        # Draw wumpus position
        if self._current_state is not None:
            wumpus_x = margin + self._current_state["x"] * cell_size + cell_size // 2
            wumpus_y = margin + self._current_state["y"] * cell_size + cell_size // 2
            wumpus_radius = cell_size // 4
            pygame.draw.circle(self.surf, WUMPUS_COLOR, (wumpus_x, wumpus_y), wumpus_radius)
            pygame.draw.circle(self.surf, TEXT_COLOR, (wumpus_x, wumpus_y), wumpus_radius, 2)

        # Draw labels
        font = pygame.font.Font(None, 24)
        
        # Gold label
        gold_text = font.render("GOLD", True, TEXT_COLOR)
        gold_text_rect = gold_text.get_rect(center=(gold_x, gold_y - gold_radius - 15))
        self.surf.blit(gold_text, gold_text_rect)
        
        # Wumpus label
        if self._current_state is not None:
            wumpus_text = font.render("WUMPUS", True, TEXT_COLOR)
            wumpus_text_rect = wumpus_text.get_rect(center=(wumpus_x, wumpus_y - wumpus_radius - 15))
            self.surf.blit(wumpus_text, wumpus_text_rect)

        # Draw coordinates
        coord_font = pygame.font.Font(None, 16)
        for i in range(self.size):
            for j in range(self.size):
                coord_text = coord_font.render(f"({i},{j})", True, TEXT_COLOR)
                coord_x = margin + i * cell_size + 5
                coord_y = margin + j * cell_size + 5
                self.surf.blit(coord_text, (coord_x, coord_y))

        self.screen.blit(self.surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(20)  # 20 FPS for wumpus world
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))






