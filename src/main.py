import os
import pygame

from const import FPS, EPISODES, MAX_ACTIONS
from environment import UnknownAngryBirds, PygameInit
from analysis import save_plot, plot_policy_grid
from agent import QLearningAgent


def main():
    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()

    action_space = 4  # Up, Down, Left, Right
    agent = QLearningAgent(action_space)

    episode_rewards = []
    value_differences = []
    final_pig_state = None

    for episode in range(EPISODES):
        state = env.reset()
        pig_state = tuple([True] * 8)
        total_reward = 0
        value_diff = 0

        for step in range(MAX_ACTIONS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            action = agent.choose_action(pig_state, state)
            next_state, reward, next_pig_state, done = env.step(action)
            next_pig_state = tuple(next_pig_state)

            prev_q_value = agent.q_table[pig_state][state].get(action, 0.0)
            agent.update_q_value(pig_state, state, action, reward, next_pig_state, next_state, done)

            # Track value difference
            current_q_value = agent.q_table[pig_state][state].get(action, 0.0)
            value_diff += abs(current_q_value - prev_q_value)

            state = next_state
            pig_state = next_pig_state
            total_reward += reward

            if done:
                break

            env.render(screen)
            pygame.display.flip()
            clock.tick(FPS)

        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}, Value Difference: {value_diff}")

        episode_rewards.append(total_reward)
        value_differences.append(value_diff)
        final_pig_state = pig_state

    print(f'MEAN REWARD: {sum(episode_rewards) / len(episode_rewards)}')

    save_plot("Reward", episode_rewards, "../plots/rewards_per_episode.png")
    save_plot("Value Difference", value_differences, "../plots/value_differences_per_episode.png")
    plot_policy_grid(agent.q_table, final_pig_state)

    pygame.quit()


if __name__ == "__main__":
    os.makedirs("../plots", exist_ok=True)
    main()
