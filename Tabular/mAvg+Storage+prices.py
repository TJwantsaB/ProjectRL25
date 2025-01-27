import numpy as np
import random
import gym
from env import DataCenterEnv  # Make sure this import points to your env.py file
from matplotlib import pyplot as plt
from collections import deque

class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=0.99,
        bin_size_storage=15,   # A bit bigger than 5
        bin_size_price=5,     # A bit bigger than 5
        bin_size_hour=24,      # One bin per hour is convenient
        bin_size_day=1,
        episodes=100,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9,
        rolling_window_size=72,
        storage_factor=1,
        min_max_price=20
    ):
        """
        Q-learning agent for the DataCenterEnv.

        The biggest fix we need is to ensure we properly reset the environment
        each episode, because the environment doesn't have a built-in reset() method.
        """
        self.env = environment
        self.discount_rate = discount_rate
        self.bin_size_storage = bin_size_storage
        self.bin_size_price = bin_size_price
        self.bin_size_hour = bin_size_hour
        self.bin_size_day = bin_size_day

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.price_history = deque(maxlen=rolling_window_size)
        self.min_max_price = min_max_price

        # Define ranges for discretization.
        # You can tune these if you have reason to believe the datacenter might
        # store more or less than 170 MWh, or see higher/lower prices, etc.
        self.storage_min = 0.0
        self.storage_max = 150
        self.storage_factor = storage_factor
        self.price_min = -self.min_max_price
        self.price_max = self.min_max_price
        # Hour range is integer 1..24. We'll create 24 bins so each hour is its own bin.
        self.hour_min = 1
        self.hour_max = 24
        # Day range. We can do day modulo 7 or something. We'll do that in `discretize_state`.
        self.day_min = 1
        self.day_max = 7

        # Create bin edges.
        self.bin_storage_edges = np.linspace(
            self.storage_min, self.storage_max, self.bin_size_storage + 1
        )

        self.bin_price_edges = np.linspace(
            self.price_min, self.price_max, self.bin_size_price
        )
        self.bin_hour_edges = np.linspace(
            self.hour_min - 0.5, self.hour_max + 0.5, self.bin_size_hour + 1
        )

        self.bin_day_edges = np.linspace(
            self.day_min - 0.5,
            self.day_min + self.bin_size_day - 0.5,
            self.bin_size_day
        )

        # Discretize the action space. We'll have 5 possible actions in [-1, -0.5, 0, 0.5, 1].
        self.discrete_actions = [-1,0,1]
        self.action_size = len(self.discrete_actions)

        # Create Q-table: shape = [storage_bins, price_bins, hour_bins, day_bins, action_size]
        self.Q_table = np.full(
            (
                self.bin_size_storage,
                self.bin_size_price,
                self.bin_size_hour,
                self.bin_size_day,
                self.action_size
            ),100
        )

        # For logging
        self.episode_rewards = []
        self.average_rewards = []

    def discretize_state(self, state_raw):
        """
        Convert continuous state [storage_level, price, hour, day]
        into discrete indices for each dimension.
        """
        storage_level, price, hour, day = state_raw

        # We can do day modulo bin_size_day if we want a repeating pattern:
        day_mod = (day - 1) % self.bin_size_day + 1

        idx_storage = np.digitize(storage_level, self.bin_storage_edges) - 1
        idx_storage = np.clip(idx_storage, 0, self.bin_size_storage - 1)

        if len(self.price_history) > 0:
            relative_price = price - np.median(self.price_history)
        else:
            relative_price = 0

        idx_price = np.digitize(relative_price, self.bin_price_edges) - 1
        idx_price = np.clip(idx_price, 0, self.bin_size_price - 1)

        idx_hour = np.digitize(hour, self.bin_hour_edges) - 1
        idx_hour = np.clip(idx_hour, 0, self.bin_size_hour - 1)

        idx_day = np.digitize(day_mod, self.bin_day_edges) - 1
        idx_day = np.clip(idx_day, 0, self.bin_size_day - 1)

        return (idx_storage, idx_price, idx_hour, idx_day)

    def epsilon_greedy_action(self, state_disc):
        """
        Pick an action index using epsilon-greedy policy, ensuring 'sell' is not selected if storage is empty.
        """
        storage_level_idx = state_disc[0]  # Storage index

        # If storage is empty, disallow the "sell" action
        if storage_level_idx == 0:
            valid_actions = [1, 2]  # Only "do nothing" and "buy"
            valid_probs = [0.5, 0.5]  # Adjusted probabilities
        else:
            valid_actions = [0, 1, 2]  # All actions are valid
            valid_probs = [0.25, 0.25, 0.5]  # Default probabilities

        if random.uniform(0, 1) < self.epsilon:
            # Explore: Choose an action from valid actions
            return np.random.choice(valid_actions, p=valid_probs)
        else:
            # Exploit: Greedy selection only from valid actions
            q_values = self.Q_table[
                state_disc[0], state_disc[1], state_disc[2], state_disc[3]
            ]
            # Mask "sell" action if storage is empty
            if storage_level_idx == 0:
                q_values = np.array([float('-inf'), q_values[1], q_values[2]])

            return np.argmax(q_values)

    # Determine whether agent is forced to buy
    def force_buy(self, state_disc):
        hours_left = 24 - state_disc[2] - 1
        shortfall = 120 - ((state_disc[0]) * 10)
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
            terminated = False

            total_reward = 0.0

            while not terminated:
                # We'll then do a fresh reset next episode.
                if self.env.day >= len(self.env.price_values):
                    # This is where the environment is done for the data set
                    terminated = True
                    break

                #################################################
                #  Let agent explore states with higher energy  #
                #################################################

                # Discretize state
                state_disc = self.discretize_state(state)

                if self.force_buy(state_disc):
                    action_idx = 2
                else:
                    # Epsilon-greedy action
                    action_idx = self.epsilon_greedy_action(state_disc)

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
                shaped_reward.clip(min=-50, max=50)

                #####################################
                #  Reward agent for storing energy  #
                #####################################

                energy_proportional_reward = next_state[0] * self.storage_factor
                if state[0] >= 170:
                    energy_proportional_reward = -energy_proportional_reward
                shaped_reward += energy_proportional_reward

                # Discretize next state
                next_state_disc = self.discretize_state(next_state)

                # Q-learning update
                old_value = self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
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
                        next_state_disc[3],
                        2
                    ]

                else:
                    next_max = np.max(
                        self.Q_table[
                            next_state_disc[0],
                            next_state_disc[1],
                            next_state_disc[2],
                            next_state_disc[3]
                        ]
                    )

                td_target = shaped_reward + self.discount_rate * next_max

                new_value = old_value + self.learning_rate * (td_target - old_value)
                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
                    action_idx
                ] = new_value

                total_reward += reward
                state = next_state

                if random.uniform(0, 1) < self.epsilon:
                    if state[2] == 1:
                        self.env.storage_level = random.choice([0, 10, 20, 30, 40, 50])

            # Decay epsilon after each full episode
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

            print(total_reward)

        np.save('best_one_yet.npy', self.Q_table)
        print("Training finished!")

    def act(self, state):
        """
        Use trained Q-table (greedy) for action selection, no exploration.
        """
        state_disc = self.discretize_state(state)
        current_price = state[1]
        self.price_history.append(current_price)


        best_action_idx = np.argmax(
            self.Q_table[
                state_disc[0],
                state_disc[1],
                state_disc[2],
                state_disc[3]
            ]
        )
        return self.discrete_actions[best_action_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    # Define ranges for min_max_price and price bins
    min_max_price_range = range(10, 101, 10)  # From 10 to 100 with step size 10
    price_bins = [3, 5, 7]

    for min_max_price in min_max_price_range:
        for price_bin in price_bins:
            print(f"\nRunning training with min_max_price={min_max_price} and price_bin={price_bin}\n")

            # Create environment
            env = DataCenterEnv(path_to_test_data=args.path, price_bin=price_bin)

            # Create agent
            agent = QAgentDataCenter(
                environment=env,
                episodes=30,         # you can reduce or increase
                learning_rate=0.005,
                discount_rate=1,
                epsilon=1.0,
                epsilon_min=0.00,
                epsilon_decay=0.67,  # so we see faster decay for demo
                rolling_window_size=27,
                min_max_price=min_max_price
            )

            # Train
            agent.train()

            # Test run with the greedy policy
            print("\nRunning a quick greedy run with the learned policy:")

            env = DataCenterEnv(path_to_test_data='validate.xlsx', price_bin=price_bin)

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

            print(f"Total reward using the greedy policy after training: {total_greedy_reward:.2f}\n")
