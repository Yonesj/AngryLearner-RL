# **Angry Bird agent: Reinforcement Learning in a Grid Environment**

This project creates a simple simulation environment inspired by a grid-based version of **Angry Birds**. In this game-like setting, a "yellow bird" (the agent) learns to move across an **8x8 grid** from the top-left corner `(0, 0)` to the bottom-right corner `(7, 7)`, where a "blue bird" awaits. Along the way, the agent encounters different objects, such as pigs or rocks, that provide rewards or penalties based on type of object. Additionally, the agent's movements are not always predictable, as random factors cause it to move differently than intended which adds more complexity to navigation problem. This project uses the **Q-learning** algorithm to teach the agent how to make better decisions over time.

<br>

https://github.com/user-attachments/assets/4c07e11e-5f3b-4248-a8fc-c9ff9f21db11

<br>

## **Environment**

The environment includes the following attributes:

- **Grid Size**: 8x8 grid.

- **Objects**:
  - **Pigs**: +250 reward upon interaction.
  - **Queen Pigs**: -400 penalty upon interaction.
  - **TNT**: -2000 penalty upon interaction.
  - **Rocks**: Block movement, acting as obstacles.
  
- **Stochastic Behavior**: The agent can move up, down, left, or right, but due to stochasticity, it may not always move in the intended direction.

<br>

## **How to Run**

1. **Setup**: Ensure all dependencies are installed.

   ```bash
   pip install -r requirements.txt
   ```   
   
2. **Run the Simulation**: Execute the main script to start the agent learning process:

   ```bash
   python main.py
   ```

3. **Monitor Progress**:  
   - The agent's policy will be visualized on the grid.  
   - Value difference charts will be plotted to track convergence over episodes.

<br>

## **Features**

- **Environment Setup**:
  Randomly generated 8x8 grid with obstacles, pigs, queen pigs, and TNT.

- **Agent**:
  Implements reinforcement learning algorithms to learn the optimal policy.

- **Visualization Tools**:
  Plot policy maps and value difference charts to analyze learning progress and convergence.

<br>

## **Source Files**

```plaintext
.
├── agent.py          # Implements the learning agent (Q-Learning)
├── const.py          # Contains constants like rewards, colors, and actions
├── environment.py    # Defines the game environment, grid setup, and step mechanics
├── main.py           # Runs the simulation and integrates agent with the environment
└── analysis.py       # Provides functions to visualize policy and learning metrics
```

<br>

## **Learning Convergence**

To verify learning convergence:

- The **Value Difference** is computed after each episode to measure Q-table updates:

- A decreasing Value Difference indicates convergence. Learning stops when the value difference becomes sufficiently small.

<br>

## **License**

This project is licensed under the Apache License 2.0. You are free to use, modify, and distribute the code, provided proper attribution is given and the terms of the license are followed. See the [LICENSE](LICENSE) file for more details.

