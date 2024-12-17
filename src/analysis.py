# Plot results
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


def save_plot(plot_name: str, items: List[float], file_path: str) -> None:
    """Save the plot as an image file."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(items) + 1), items, marker='o')
    plt.title(f"{plot_name}s per Episode")
    plt.xlabel("Episode")
    plt.ylabel(plot_name)
    plt.grid(True)
    plt.savefig(file_path, format='png')
    plt.close()


def plot_policy_grid(q_table: Dict[Tuple[bool], Dict[Tuple[int, int], Dict[int, float]]],
                     pig_state: Tuple[bool],
                     grid_size: Tuple[int, int] = (8, 8),
                     output_path: str = "../plots/policy_grid.png") -> None:
    """
    Plot an 8x8 grid showing the best action for each state based on the policy for a specific pig state.

    Parameters:
    q_table: Dict[Tuple[bool], Dict[Tuple[int, int], Dict[int, float]]]
        The Q-table where each pig state, state, and action pair has a value.
    pig_state: tuple
        The pig state to use when generating the policy grid.
    grid_size: tuple
        The dimensions of the grid (default is 8x8).
    output_path: str
        The file path to save the policy grid plot.
    """
    actions = ["↑", "→", "↓", "←"]  # Represent actions as arrows
    policy_grid = np.empty(grid_size, dtype=object)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            action_values = [q_table[pig_state][state][action] for action in range(len(actions))]
            best_action = int(np.argmax(action_values))  # Get the action with the highest Q-value
            policy_grid[i, j] = actions[best_action]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax.text(j + 0.5, grid_size[0] - i - 0.5, policy_grid[i, j],
                    ha='center', va='center', fontsize=12)

    # Draw grid lines
    for i in range(grid_size[0] + 1):
        ax.axhline(i, color='black', linewidth=0.5)
    for j in range(grid_size[1] + 1):
        ax.axvline(j, color='black', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Policy Grid for Pig State: {pig_state}", fontsize=16)
    plt.savefig(output_path)
    plt.close()

