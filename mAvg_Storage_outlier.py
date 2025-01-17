import numpy as np
import random
import gym
from env import DataCenterEnv  # Make sure this import points to your env.py file
from matplotlib import pyplot as plt
from collections import deque
import os
import imageio
import wandb

class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=0.99,
        bin_size_storage=12,
        bin_size_hour=24,
        bin_size_outlier=3,
        episodes=100,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9,
        rolling_window_size=72,
        storage_factor=0.5,
        wandb=False,
    ):
        """
        Q-learning agent for the DataCenterEnv.

        The biggest fix we need is to ensure we properly reset the environment
        each episode, because the environment doesn't have a built-in reset() method.
        """
        self.env = environment
        self.discount_rate = discount_rate
        self.bin_size_storage = bin_size_storage
        self.bin_size_hour = bin_size_hour
        self.bin_size_outlier = bin_size_outlier

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.price_history = deque(maxlen=rolling_window_size)

        self.storage_min = 0.0
        self.storage_max = 120
        self.storage_factor = storage_factor

        self.hour_min = 1
        self.hour_max = 24

        self.action_counts = np.zeros(3)  # Counts of actions taken
        self.action_rewards = np.zeros(3)  # Sum of rewards for each action
        self.action_means = np.zeros(3)  # Mean rewards for each action


        # Create bin edges.
        self.bin_storage_edges = np.linspace(
            self.storage_min, self.storage_max, self.bin_size_storage
        )
        self.bin_hour_edges = np.linspace(
            self.hour_min - 0.5, self.hour_max + 0.5, self.bin_size_hour
        )

        self.bin_outlier_edges = [-np.inf, -1.5, 1.5, np.inf]

        # Discretize the action space. We'll have 5 possible actions in [-1, -0.5, 0, 0.5, 1].
        self.discrete_actions = [-1,0,1]
        self.action_size = len(self.discrete_actions)

        # Create Q-table: shape = [storage_bins, price_bins, hour_bins, day_bins, action_size]
        self.Q_table = np.full(
            (
                self.bin_size_storage,
                self.bin_size_hour,
                self.bin_size_outlier,
                self.action_size
            ), 0.01
        )

        # For logging
        self.episode_rewards = []
        self.average_rewards = []
        self.outlier_log = {i: [] for i in range(self.bin_size_outlier)}

        self.wandb = wandb

    def discretize_state(self, state_raw):
        """
        Convert continuous state [storage_level, price, hour, day]
        into discrete indices for each dimension.
        """
        storage_level, price, hour, day = state_raw

        idx_storage = np.digitize(storage_level, self.bin_storage_edges) - 1
        idx_storage = np.clip(idx_storage, 0, self.bin_size_storage - 1)

        idx_hour = np.digitize(hour, self.bin_hour_edges) - 1
        idx_hour = np.clip(idx_hour, 0, self.bin_size_hour - 1)

        # Safeguard z-score calculation
        if len(self.price_history) > 0 and np.std(self.price_history) > 0:
            z_score = (price - np.mean(self.price_history)) / np.std(self.price_history)
        else:
            z_score = 0

        idx_outlier = np.digitize(z_score, self.bin_outlier_edges) - 1
        idx_outlier = np.clip(idx_outlier, 0, self.bin_size_outlier - 1)

        return (idx_storage, idx_hour, idx_outlier)

    def epsilon_greedy_action(self, state_disc):
        """
        Pick an action index using epsilon-greedy policy with state-dependent biases.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Sample action based on biased probabilities
            probabilities = [0.15, 0.15, 0.70]
            actions = [0, 1, 2]  # 0 -> sell, 1 -> do nothing, 2 -> buy
            return np.random.choice(actions, p=probabilities)

        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(
                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2]
                ]
            )

    def thompson_sampling_action(self, state_disc):
        """
        Select an action using Thompson Sampling.
        """
        sampled_rewards = []
        for action_idx in range(self.action_size):
            # Mean and variance of reward distribution for the action
            mean = self.action_means[action_idx]
            variance = 1 / (self.action_counts[action_idx] + 1)  # Add 1 to avoid division by zero

            # Sample a reward from the Gaussian distribution
            sampled_reward = np.random.normal(mean, np.sqrt(variance))
            sampled_rewards.append(sampled_reward)

        # Choose the action with the highest sampled reward
        return np.argmax(sampled_rewards)

    # Determine whether agent is forced to buy
    def force_buy(self, state_disc):
        hours_left = 24 - state_disc[1]
        shortfall = 120 - (state_disc[0]+1 * 10)
        max_possibly_buy = hours_left * 10
        return shortfall > max_possibly_buy

    def _manual_env_reset(self):
        """
        Because env.py has no 'reset' method, we manually reset
        day, hour, and storage_level to start a new episode:
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        # Then call observation() to get the fresh initial state
        return self.env.observation()

    def train(self):
        """
        Train the agent over a number of episodes.
        """

        for episode in range(self.episodes):
            print(f"Episode {episode + 1}")

            # Manually reset environment at start of each episode
            state = self._manual_env_reset()
            self.price_history.clear()
            terminated = False

            total_reward = 0.0
            total_td_error = 0

            steps = 0

            while not terminated:
                # Check if the environment is out of data
                if self.env.day >= len(self.env.price_values):
                    terminated = True
                    break

                # Discretize state
                state_disc = self.discretize_state(state)

                if self.force_buy(state_disc):
                    action_idx = 2
                else:
                    action_idx = self.epsilon_greedy_action(state_disc)
                    chosen_action = self.discrete_actions[action_idx]

                steps += 1

                #################################################
                #  Let agent explore states with higher energy  #
                #################################################

                if random.uniform(0, 1) < self.epsilon:
                    if state[2] == 1:
                        state[0] = 50


                # Discretize state
                state_disc = self.discretize_state(state)


                if self.force_buy(state_disc):
                    action_idx = 2
                else:
                    action_idx = self.thompson_sampling_action(state_disc)

                chosen_action = self.discrete_actions[action_idx]

                # Step environment
                next_state, reward, terminated = self.env.step(chosen_action)

                ########################################################################
                #  Keep track of rolling average of prices and reward proportionately  #
                #  Allows agent to get positive reward for buying at low prices        #
                ########################################################################

                # Compute reward relative to moving average
                def rolling_reward(chosen_action, reward, rolling_avg_price):
                    return rolling_avg_price * 10 * chosen_action + reward

                current_price = state[1]
                self.price_history.append(current_price)
                rolling_avg_price = np.mean(self.price_history)

                shaped_reward =  rolling_reward(chosen_action, reward, rolling_avg_price)

                self.outlier_log[state_disc[2]].append((chosen_action, shaped_reward))

                #####################################
                #  Reward agent for storing energy  #
                #####################################

                energy_proportional_reward = next_state[0] * self.storage_factor
                shaped_reward += energy_proportional_reward

                shaped_reward = shaped_reward / 100
                shaped_reward = np.clip(shaped_reward, -100, 100)  # Clipping to the observed range

                # Update action counts and rewards
                self.action_counts[action_idx] += 1
                self.action_rewards[action_idx] += shaped_reward
                self.action_means[action_idx] = self.action_rewards[action_idx] / self.action_counts[action_idx]

                # Discretize next state
                next_state_disc = self.discretize_state(next_state)

                # Q-learning update
                old_value = self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    action_idx
                ]

                #########################################################
                #  Check whether agent if forced to buy, if so:         #
                #  Q_value for update should be that of buy (action 2)  #
                #########################################################

                if self.force_buy(state_disc):
                    next_max = self.Q_table[
                        next_state_disc[0],
                        next_state_disc[1],
                        next_state_disc[2],
                        2
                    ]

                else:
                    next_max = np.max(
                        self.Q_table[
                            next_state_disc[0],
                            next_state_disc[1],
                            next_state_disc[2]
                        ]
                    )

                td_target = shaped_reward + self.discount_rate * next_max
                td_error = td_target - old_value
                total_td_error += abs(td_error)
                new_value = old_value + self.learning_rate * td_error

                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    action_idx
                ] = new_value

                total_reward += reward
                state = next_state


            avg_td_error = total_td_error / steps
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
            #log episode, reward entropy
            if (episode + 1) % 10 == 0:
                if self.wandb:
                    wandb.log({
                        "episode": episode + 1,
                        "epsilon": self.epsilon,
                        "total_reward": total_reward,
                        "avg_td_error": avg_td_error
                    })

            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.episode_rewards.append(total_reward)
            # Print average every 50 episodes
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.average_rewards.append(avg_reward)
                print(
                    f"Episode {episode + 1}, "
                    f"Avg reward (last 50): {avg_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )


        np.save('final_q_table_YEAH.npy', self.Q_table)
        self.analyze_outlier_usage()

        print("Training finished!")


    def act(self, state):
        """
        Use trained Q-table (greedy) for action selection, no exploration.
        """
        state_disc = self.discretize_state(state)
        best_action_idx = np.argmax(
            self.Q_table[
                state_disc[0],
                state_disc[1],
                state_disc[2]
            ]
        )
        return self.discrete_actions[best_action_idx]

    def analyze_outlier_usage(self):
        """
        Analyze and visualize how the agent uses the idx_outlier dimension, and log the results to W&B.
        """
        outlier_data = {}

        print("\nOutlier Usage Analysis:")
        for outlier_idx, logs in self.outlier_log.items():
            actions, rewards = zip(*logs) if logs else ([], [])
            total_occurrences = len(logs)
            action_distribution = {action: count for action, count in zip(*np.unique(actions, return_counts=True))}
            avg_reward = np.mean(rewards) if rewards else 0

            # Print the results to console
            print(f"Outlier Index {outlier_idx}:")
            print(f"  Total Occurrences: {total_occurrences}")
            print(f"  Action Distribution: {action_distribution}")
            print(f"  Average Reward: {avg_reward:.2f}\n")

            # Prepare data for W&B logging
            outlier_data[outlier_idx] = {
                "total_occurrences": total_occurrences,
                "action_distribution": action_distribution,
                "avg_reward": avg_reward
            }

        # Log to W&B
        if self.wandb:
            for outlier_idx, data in outlier_data.items():
                wandb.log({
                    f"outlier_{outlier_idx}_total_occurrences": data["total_occurrences"],
                    f"outlier_{outlier_idx}_action_distribution": data["action_distribution"],
                    f"outlier_{outlier_idx}_avg_reward": data["avg_reward"]
                })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    # Create environment
    env = DataCenterEnv(path_to_test_data=args.path)

    # Create agent
    agent = QAgentDataCenter(
        environment=env,
        episodes=300,         # you can reduce or increase
        learning_rate=0.1,
        discount_rate=0.90,
        epsilon=1.0,
        epsilon_min=0.10,
        epsilon_decay=0.965,  # so we see faster decay for demo
        rolling_window_size=12

    )

    # Train
    agent.train()

    # Test run with the greedy policy
    print("\nRunning a quick greedy run with the learned policy:")

    # We do a fresh manual reset for the test run:
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    state = env.observation()
    terminated = False
    total_greedy_reward = 0.0

    while not terminated:
        if env.day >= len(env.price_values):
            break
        action = agent.act(state)
        next_state, reward, terminated = env.step(action)
        total_greedy_reward += reward
        state = next_state
        print("Action:", action)
        print("Next state:", next_state)
        print("Reward:", reward)

    print(f"Total reward using the greedy policy after training: {total_greedy_reward:.2f}")
