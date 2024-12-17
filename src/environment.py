import random
import copy
from typing import List, Tuple, Dict

import numpy as np
import pygame

from const import *


class PygameInit:
    """
    Initializes Pygame with a specified grid and tile size.

    Methods:
        initialization():
            Sets up the Pygame window and returns the screen and clock.
    """

    @classmethod
    def initialization(cls) -> Tuple[pygame.Surface, pygame.time.Clock]:
        """
        Initializes Pygame screen and clock.

        Returns:
            Tuple[pygame.Surface, pygame.time.Clock]: Pygame screen and clock objects.
        """
        grid_size = 8
        tile_size = 100

        pygame.init()
        screen = pygame.display.set_mode((grid_size * tile_size, grid_size * tile_size))
        pygame.display.set_caption("Unknown Angry Birds")
        clock = pygame.time.Clock()

        return screen, clock


class UnknownAngryBirds:
    """
    Represents the Unknown Angry Birds game environment.

    Attributes:
        actions (dict): Possible actions the agent can take.
        neighbors (dict): Neighboring actions for probabilistic movement.

    Methods:
        reset():
            Resets the environment to its initial state.
        step(action):
            Executes an action and returns the next state, reward, pigs and termination status.
        render(screen):
            Renders the environment grid and agent on the given Pygame screen.
    """
    actions = ACTIONS
    neighbors = NEIGHBORS

    def __init__(self):
        """
        Initializes the Angry Birds environment with default parameters and assets.
        """
        self.__grid_size = 8
        self.__tile_size = 100
        self.__num_pigs = PIGS
        self.__num_queens = QUEENS
        self.__num_rocks = ROCKS
        self.__num_tnts = TNTs

        self.reward = 0
        self.done = False
        self.pig_states = []

        self.__pig_coordinates = []
        self.__base_grid = self.__generate_grid()
        self.__grid = copy.deepcopy(self.__base_grid)
        self.__probability_dict = self.__generate_probability_dict()

        self.__agent_pos = (0, 0)
        self.__max_actions = MAX_ACTIONS
        self.__actions_taken = 0

        self.__load_assets()

    def __load_assets(self):
        """
        Loads and processes image assets for the game.
        """
        self.__agent_image, _ = self.__load_and_scale_image("../assets/yellow bird.png")
        self.__pig_image, self.__pig_with_background = self.__load_and_scale_image('../assets/pigs.png', True)
        self.__blue_bird, self.__blue_bird_with_background = self.__load_and_scale_image("../assets/angry bird blue.png", True)
        self.__queen_image, self.__queen_with_background = self.__load_and_scale_image('../assets/queen.png', True)
        self.__rock_image, self.__rock_with_background = self.__load_and_scale_image('../assets/rocks.png', True)
        self.__tnt_image, self.__tnt_background = self.__load_and_scale_image("../assets/TNT.png", True)

    def __load_and_scale_image(self, path: str, with_background: bool = False) -> Tuple[pygame.Surface, pygame.Surface]:
        """
        Loads and scales an image, optionally applying a background.

        Args:
            path (str): Path to the image file.
            with_background (bool): Whether to apply a background.

        Returns:
            pygame.Surface: The processed image surface.
        """
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (self.__tile_size, self.__tile_size))
        surface = None

        if with_background:
            surface = pygame.Surface((self.__tile_size, self.__tile_size))
            surface.fill((245, 245, 220))
            surface.blit(image, (0, 0))

        return image, surface

    def __generate_grid(self) -> List[List[str]]:
        """
        Generates the initial grid for the environment.

        Returns:
            List[List[str]]: A 2D list representing the grid.
        """
        while True:
            grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]
            filled_spaces = [(0, 0), (self.__grid_size - 1, self.__grid_size - 1)]

            self.__populate_grid(grid, 'P', self.__num_pigs, filled_spaces)
            self.__populate_grid(grid, 'Q', self.__num_queens, filled_spaces)
            self.__populate_grid(grid, 'R', self.__num_rocks, filled_spaces)
            self.__populate_grid(grid, "TNT", self.__num_tnts, filled_spaces)

            grid[self.__grid_size - 1][self.__grid_size - 1] = 'G'

            if UnknownAngryBirds.__is_path_exists(grid=grid, start=(0, 0), goal=(7, 7)):
                break

        return grid

    def __populate_grid(self, grid: List[List[str]], symbol: str, count: int, filled_spaces: List[Tuple[int, int]]) -> None:
        """
        Populates the grid with a specific symbol at random locations.

        Args:
            grid (List[List[str]]): The grid to populate.
            symbol (str): The symbol to place in the grid.
            count (int): Number of symbols to place.
            filled_spaces (List[Tuple[int, int]]): List of already filled spaces.
        """
        for _ in range(count):
            while True:
                r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                if (r, c) not in filled_spaces:
                    grid[r][c] = symbol
                    filled_spaces.append((r, c))
                    break

    def reset(self) -> Tuple[int, int]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[int, int]: The agent's initial position.
        """
        self.__grid = copy.deepcopy(self.__base_grid)
        self.__agent_pos = (0, 0)
        self.done = False
        self.__actions_taken = 0
        return self.__agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], int, List[bool], bool]:
        """
        Executes the given action and returns the next state, reward, pig states and termination status.

        Args:
            action (int): The action to execute (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            Tuple[Tuple[int, int], int, List[bool], bool]:
                - Next state (row, col)
                - Reward received
                - pigs state
                - Termination status
        """
        prob_dist = self.__get_action_probability_distribution(action)
        chosen_action = np.random.choice([0, 1, 2, 3], p=prob_dist)

        dx, dy = self.actions[chosen_action]
        new_row, new_col = self.__agent_pos[0] + dx, self.__agent_pos[1] + dy

        if (0 <= new_row < self.__grid_size and 0 <= new_col < self.__grid_size and
                self.__grid[new_row][new_col] != 'R'):
            self.__agent_pos = (new_row, new_col)

        self.__actions_taken += 1
        current_tile = self.__grid[self.__agent_pos[0]][self.__agent_pos[1]]
        reward = DEFAULT_REWARD

        if current_tile == 'Q':
            reward = QUEEN_REWARD
            self.__grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'P':
            reward = GOOD_PIG_REWARD
            self.__grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'G':
            reward = GOAL_REWARD
            self.done = True

        elif current_tile == 'TNT':
            reward = TNT_REWARD
            self.done = True

        elif current_tile == 'T':
            reward = DEFAULT_REWARD

        if self.__actions_taken >= self.__max_actions:
            reward = ACTION_TAKEN_REWARD
            self.done = True

        self.reward = reward
        return self.__agent_pos, self.reward, self.__get_pig_state(), self.done

    def __get_action_probability_distribution(self, action: int) -> List[float]:
        """
        Gets the probability distribution of possible actions based on the intended action.

        Args:
            action (int): The intended action index (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            List[float]: A list representing the probability distribution of actions.
        """
        prob_dist = [0] * 4
        prob_dist[action] = self.__probability_dict[self.__agent_pos][action]['intended']

        for neighbor_action in self.neighbors[action]:
            prob_dist[neighbor_action] = self.__probability_dict[self.__agent_pos][action]['neighbor']

        return prob_dist

    def render(self, screen: pygame.Surface) -> None:
        """
        Renders the environment grid and agent on the given Pygame screen.

        Args:
            screen (pygame.Surface): The Pygame screen to render on.
        """
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = COLORS[self.__grid[r][c]]
                pygame.draw.rect(screen, color, (c * self.__tile_size, r * self.__tile_size, self.__tile_size,
                                                 self.__tile_size))

                self.__render_tile(screen, r, c)

        self.__draw_grid_lines(screen)
        agent_row, agent_col = self.__agent_pos
        screen.blit(self.__agent_image, (agent_col * self.__tile_size, agent_row * self.__tile_size))

    def __render_tile(self, screen: pygame.Surface, row: int, col: int) -> None:
        """
        Renders specific tiles based on their type.

        Args:
            screen (pygame.Surface): The Pygame screen.
            row (int): Row index.
            col (int): Column index.
        """
        tile = self.__grid[row][col]
        tile_mapping = {
            'P': self.__pig_with_background,
            'G': self.__blue_bird_with_background,
            'Q': self.__queen_with_background,
            'R': self.__rock_with_background,
            'TNT': self.__tnt_background
        }

        if tile in tile_mapping:
            screen.blit(tile_mapping[tile], (col * self.__tile_size, row * self.__tile_size))

    def __draw_grid_lines(self, screen: pygame.Surface) -> None:
        """
        Draws grid lines on the Pygame screen.

        Args:
            screen (pygame.Surface): The Pygame screen.
        """
        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size),
                             (self.__grid_size * self.__tile_size, r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0),
                             (c * self.__tile_size, self.__grid_size * self.__tile_size), 2)

    @classmethod
    def __is_path_exists(cls, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Checks if a valid path exists from the start position to the goal position.

        Args:
            grid (List[List[str]]): The grid representation of the environment.
            start (Tuple[int, int]): The starting position.
            goal (Tuple[int, int]): The goal position.

        Returns:
            bool: True if a valid path exists, False otherwise.
        """
        grid_size = len(grid)
        visited = set()

        def dfs(x, y):
            if (x, y) == goal:
                return True
            visited.add((x, y))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size and 0 <= ny < grid_size and
                        (nx, ny) not in visited and grid[nx][ny] != 'R'):
                    if dfs(nx, ny):
                        return True
            return False

        return dfs(start[0], start[1])

    def __generate_probability_dict(self) -> Dict[Tuple[int, int], Dict[int, dict]]:
        """
        Generates a probability distribution for each state and action.

        Returns:
            dict: A nested dictionary where keys are states (row, col) and actions.
                  Each action contains probabilities for the intended action and neighboring actions.
        """
        probability_dict = {}

        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                state = (row, col)
                probability_dict[state] = {}

                for action in range(4):
                    intended_prob = random.uniform(0.55, 0.90)
                    remaining_prob = 1 - intended_prob
                    neighbor_prob = remaining_prob / 2

                    probability_dict[state][action] = {
                        'intended': intended_prob,
                        'neighbor': neighbor_prob
                    }

        return probability_dict

    def __get_pig_state(self) -> List[bool]:
        """
        Retrieves the current states of all pigs on the grid.

        Returns:
            List[bool]: A list of booleans indicating whether each pig is still on the grid.
                        True means the pig is still present; False means it has been removed.
        """
        grid = self.__grid
        states = [False for _ in range(self.__num_pigs)]

        for i, pig_coordinate in enumerate(self.__pig_coordinates):
            x, y = pig_coordinate[0], pig_coordinate[1]
            if grid[x][y] == 'P':
                states[i] = True

        return states
