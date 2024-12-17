# Constants used in the Angry Birds environment.

# Frames per second for the game loop, controlling the speed of the game.
FPS = 1000

# Simulation parameters.
EPISODES = 4000          # Total number of episodes to run the environment.
MAX_ACTIONS = 150        # Maximum number of actions the agent can take in a single episode.

# Colors for different grid tiles, represented as RGB values.
COLORS = {
    'T': (245, 245, 220),    # Tile ground: beige color.
    'P': (245, 245, 220),    # Pigs: same as tile ground.
    'Q': (245, 245, 220),    # Queen pig: same as tile ground.
    'G': (245, 245, 220),    # Goal tile: where the agent wins the game.
    'R': (245, 245, 220),    # Rock: obstacle tiles.
    'TNT': (245, 245, 220),  # TNT: explosive penalty tiles.
}

# Reward values for different grid events and tile types.
GOOD_PIG_REWARD = 250     # Reward for capturing a pig.
GOAL_REWARD = 400         # Reward for reaching the goal tile (egg).
QUEEN_REWARD = -400       # Penalty for encountering a queen pig.
DEFAULT_REWARD = -1       # Default step penalty for moving to an empty tile.
TNT_REWARD = -2000        # Heavy penalty for stepping on a TNT tile.
ACTION_TAKEN_REWARD = -1000  # Penalty if the agent exceeds the maximum allowed actions.

# Number of specific objects placed in the grid.
PIGS = 8       # Total number of pigs in the environment.
QUEENS = 2     # Total number of queen pigs in the environment.
ROCKS = 8      # Total number of rocks (obstacles) in the grid.
TNTs = 1       # Total number of TNT tiles in the grid.

# Definitions of movement actions, represented as row-column changes.
ACTIONS = [
    (-1, 0),  # Up: Move one tile upward (row decreases).
    (1, 0),   # Down: Move one tile downward (row increases).
    (0, -1),  # Left: Move one tile to the left (column decreases).
    (0, 1)    # Right: Move one tile to the right (column increases).
]

# Neighbors define unintended movements based on the primary action.
# For each action, neighbors are possible alternative moves due to stochasticity.
NEIGHBORS = {
    0: [2, 3],  # Up -> Left and Right (agent may drift horizontally).
    1: [2, 3],  # Down -> Left and Right.
    2: [0, 1],  # Left -> Up and Down (agent may drift vertically).
    3: [0, 1]   # Right -> Up and Down.
}
