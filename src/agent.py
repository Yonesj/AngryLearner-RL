from typing import Tuple, Dict
import numpy as np


class QLearningAgent:
    def __init__(self, action_space: int, gamma=0.9, alpha: float = 0.05, Ne: int = 5, Rplus: float = 2.0):
        self.actions = list(range(action_space))  # Available actions in the environment
        self.gamma = gamma  # Discount factor for future rewards
        self.alpha = alpha
        self.Ne = Ne  # Exploration threshold: encourages exploring less-visited actions
        self.Rplus = Rplus  # Reward for unexplored actions

        # Q-values: q_table[pig_state][state][action]
        self.q_table: Dict[Tuple[bool], Dict[Tuple[int, int], Dict[int, float]]] = {}

        # Visit counts: n_table[pig_state][state][action]
        self.n_table: Dict[Tuple[bool], Dict[Tuple[int, int], Dict[int, int]]] = {}

        self.initialize_tables()

    def initialize_tables(self):
        """Initializes the Q-table and N-table."""
        for pig_state_tuple in [tuple(bool(i & (1 << j)) for j in range(8)) for i in range(256)]:
            self.q_table[pig_state_tuple] = {}
            self.n_table[pig_state_tuple] = {}
            for x in range(8):
                for y in range(8):
                    self.q_table[pig_state_tuple][(x, y)] = {action: 0.0 for action in self.actions}
                    self.n_table[pig_state_tuple][(x, y)] = {action: 0 for action in self.actions}

    def exploration_function(self, q_value: float, visit_count: int):
        """Exploration function to prioritize less-explored actions."""
        if visit_count < self.Ne:
            return self.Rplus  # Favor unexplored actions
        return q_value  # Use the learned Q-value for sufficiently explored actions

    def choose_action(self, pig_state: Tuple[bool], state: Tuple[int, int]):
        """Choose an action based on the exploration function."""
        state_dict = self.q_table[pig_state][state]

        action_values = [
            self.exploration_function(
                state_dict.get(action, 0.0),
                self.n_table[pig_state][state].get(action, 0)
            ) for action in self.actions
        ]
        return np.argmax(action_values)  # Select the action with the highest exploration-modified value

    def update_q_value(self, pig_state: Tuple[bool], state: Tuple[int, int], action: int, reward: int,
                       next_pig_state: Tuple[bool], next_state: Tuple[int, int], done: bool):
        """Update Q-values based on the Bellman equation."""
        state_dict = self.q_table[pig_state][state]
        n_state_dict = self.n_table[pig_state][state]

        # Increment visitation count for the current state-action pair
        n_state_dict[action] = n_state_dict.get(action, 0) + 1

        # Adaptive learning rate based on visitation count
        alpha = 1.0 / (1.0 + (n_state_dict[action] / 15))

        if done:
            target = reward  # If the episode is done, target is just the immediate reward
        else:
            next_state_dict = self.q_table[next_pig_state][next_state]
            # Get the maximum Q-value for the next state across all actions
            future_rewards = [next_state_dict.get(a, 0.0) for a in self.actions]
            # Bellman equation: current reward + discounted future reward
            target = reward + self.gamma * max(future_rewards)

        # Update the Q-value for the current state-action pair
        state_dict[action] = state_dict.get(action, 0.0) + alpha * (target - state_dict.get(action, 0.0))
