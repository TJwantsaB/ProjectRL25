import numpy as np
import random
from env import DataCenterEnv
from collections import deque

class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=1.0,
        discount_min=1.0,

        bin_size_storage=29,
        bin_size_price=1,
        bin_size_hour=24,

        episodes=100,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.67,
        explore_storage=False,

        rolling_window_size=0,
        storage_factor=0,
        min_max_price=0
    ):
        """
        Q-learning agent for the DataCenterEnv.

        The biggest fix we need is to ensure we properly reset the environment
        each episode, because the environment doesn't have a built-in reset() method.
        """
        self.env = environment
        self.discount_rate = discount_rate
        self.discount_min = discount_min

        self.bin_size_storage = bin_size_storage
        self.bin_size_price = bin_size_price
        self.bin_size_hour = bin_size_hour

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.explorare_storage = explore_storage

        self.rolling_window_size = rolling_window_size
        self.price_history = deque(maxlen=rolling_window_size)
        self.min_max_price = min_max_price

        # Define ranges for discretization.
        self.storage_min = 0.0
        self.storage_max = 290
        self.hour_min = 1
        self.hour_max = 24

        self.storage_factor = storage_factor
        self.price_min = -self.min_max_price
        self.price_max = self.min_max_price

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

        # Discretize the action space.
        self.discrete_actions = [-1, 0, 1]
        self.action_size = len(self.discrete_actions)

        # Create Q-table: shape = [storage_bins, price_bins, hour_bins, action_size]
        self.Q_table = np.full(
            (
                self.bin_size_storage,
                self.bin_size_price,
                self.bin_size_hour,
                self.action_size
            ),
            100
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

        idx_storage = np.digitize(storage_level, self.bin_storage_edges) - 1
        idx_storage = np.clip(idx_storage, 0, self.bin_size_storage - 1)

        # If rolling_window_size > 0, we use the rolling average of prices
        if len(self.price_history) > 0:
            relative_price = price - np.mean(self.price_history)
        else:
            relative_price = 0

        idx_price = np.digitize(relative_price, self.bin_price_edges) - 1
        idx_price = np.clip(idx_price, 0, self.bin_size_price - 1)

        idx_hour = np.digitize(hour, self.bin_hour_edges) - 1
        idx_hour = np.clip(idx_hour, 0, self.bin_size_hour - 1)

        return (idx_storage, idx_price, idx_hour)

    def epsilon_greedy_action(self, state_disc):
        """
        Pick an action index using epsilon-greedy policy, ensuring 'sell' is not selected if storage is empty.
        """
        storage_level_idx = state_disc[0]  # Storage index

        # If storage is empty, disallow "sell" (-1) action
        if storage_level_idx == 0:
            valid_actions = [1, 2]  # Only "do nothing" (0) and "buy" (1)
            valid_probs = [0.5, 0.5]
        else:
            valid_actions = [0, 1, 2]  # All 3 actions are valid
            valid_probs = [0.25, 0.25, 0.5]

        if random.uniform(0, 1) < self.epsilon:
            # Explore: Choose an action from valid actions
            return np.random.choice(valid_actions, p=valid_probs)
        else:
            # Exploit: Greedy selection only from valid actions
            q_values = self.Q_table[state_disc[0], state_disc[1], state_disc[2]]

            if storage_level_idx == 0:
                # mask out the "sell" action => set it to -inf so it's never chosen
                q_values = np.array([float('-inf'), q_values[1], q_values[2]])

            return np.argmax(q_values)

    def force_buy(self, state_disc):
        """
        Determine whether agent is forced to buy
        if shortfall can't be met otherwise.
        """
        hours_left = 24 - state_disc[2] - 1
        shortfall = 120 - (state_disc[0] * 10)
        max_possibly_buy = hours_left * 10
        return shortfall > max_possibly_buy

    def _manual_env_reset(self):
        """
        Because env.py has no 'reset' method, we manually reset day, hour, storage_level.
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()

    def train(self):
        """
        Train the agent over a number of episodes.
        """
        rewards_per_episode = {}
        action_distribution_per_episode = {}

        for episode in range(self.episodes):
            # Initialize action distribution for this episode
            action_distribution = {hour: {action: 0 for action in self.discrete_actions} for hour in range(1, 25)}

            state = self._manual_env_reset()
            terminated = False
            total_reward = 0.0

            while not terminated:
                if self.env.day >= len(self.env.price_values):
                    terminated = True
                    break

                # Discretize state
                state_disc = self.discretize_state(state)

                if self.force_buy(state_disc):
                    action_idx = 2  # force buy = index of +1
                else:
                    # Epsilon-greedy
                    action_idx = self.epsilon_greedy_action(state_disc)

                chosen_action = self.discrete_actions[action_idx]

                current_hour = int(state[2])
                action_distribution[current_hour][chosen_action] += 1

                next_state, reward, terminated = self.env.step(chosen_action)

                # ================ NEW SELL REWARD SHAPING =================
                # Only if chosen_action == -1, we override reward with -k/p
                # rolling_window or not, doesn't matter, user wants "set if action = -1 to -k/p"
                # We'll define a small k.  We'll also read the "current price" from state[1].
                if chosen_action == -1:
                    k = 100.0
                    price_now = state[1]
                    # To avoid division by zero if price is 0 (should be positive though)
                    if price_now <= 0:
                        price_now = 1e-3
                    reward = -(k / price_now)
                # ================ END SELL REWARD SHAPING =================

                # rolling logic if user set rolling_window_size > 0
                if self.rolling_window_size > 0:
                    self.price_history.append(state[1])

                # The rest is the existing code:

                # Extra shaping logic in code? (lines about rolling_reward?)
                # The code had a 'rolling_reward()' inline function, but we override it with the new
                # reward we already set for action=-1. So let's keep everything else the same.
                # We'll apply the code that was originally there:

                def rolling_reward(chosen_action, reward_, rolling_avg_price_):
                    return rolling_avg_price_ * 10 * chosen_action + reward_

                shaped_reward = reward  # Start from the final sell/buy/do-nothing reward

                if self.rolling_window_size > 0:
                    rolling_avg_price = np.mean(self.price_history)
                    shaped_reward = rolling_reward(chosen_action, reward, rolling_avg_price)
                    shaped_reward = np.clip(shaped_reward, -50, 50)

                energy_proportional_reward = next_state[0] * self.storage_factor
                if state[0] >= 170:
                    energy_proportional_reward = -energy_proportional_reward
                shaped_reward += energy_proportional_reward

                next_state_disc = self.discretize_state(next_state)
                old_value = self.Q_table[state_disc[0], state_disc[1], state_disc[2], action_idx]

                if self.force_buy(state_disc):
                    next_max = self.Q_table[next_state_disc[0], next_state_disc[1], next_state_disc[2], 2]
                else:
                    next_max = np.max(self.Q_table[next_state_disc[0], next_state_disc[1], next_state_disc[2]])

                if state[2] == 1:
                    self.discount_rate = 1
                self.discount_rate = 1 - (1 - self.discount_min) * (state[2] - 1) / 23

                td_target = shaped_reward + self.discount_rate * next_max
                new_value = old_value + self.learning_rate * (td_target - old_value)
                self.Q_table[state_disc[0], state_disc[1], state_disc[2], action_idx] = new_value

                total_reward += reward
                state = next_state

                if self.explorare_storage:
                    if random.uniform(0, 1) < self.epsilon:
                        if state[2] == 1:
                            self.env.storage_level = random.choice([0, 10, 20, 30, 40, 50])

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Normalize action distribution to percentages
            for hour, actions in action_distribution.items():
                total_actions = sum(actions.values())
                if total_actions > 0:
                    action_distribution[hour] = {
                        action: round((count / total_actions) * 100, 0)
                        for action, count in actions.items()
                    }

            rewards_per_episode[episode] = total_reward
            action_distribution_per_episode[episode] = action_distribution

        return rewards_per_episode, action_distribution_per_episode


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
            ]
        )
        return self.discrete_actions[best_action_idx]

    def validate(self):
        env = DataCenterEnv(path_to_test_data='validate.xlsx')

        env.day = 1
        env.hour = 1
        env.storage_level = 0.0
        state = env.observation()
        terminated = False
        total_greedy_reward = 0.0

        while not terminated:
            if env.day >= len(env.price_values):
                break
            action = self.act(state)
            next_state, reward, terminated = env.step(action)
            total_greedy_reward += reward
            state = next_state
        return total_greedy_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    env = DataCenterEnv(path_to_test_data=args.path)

    agent = QAgentDataCenter(
        environment=env,
        episodes=20,
        learning_rate=0.005,
        discount_rate=1,
        discount_min=0.5,
        epsilon=1.0,
        epsilon_min=0.00,
        epsilon_decay=0.67,
        bin_size_price=1,
        rolling_window_size=27,
        storage_factor=1,
        min_max_price=50
    )

    rewards_per_episode, action_distribution_per_episode = agent.train()

    best_episode = max(rewards_per_episode, key=rewards_per_episode.get)
    best_reward = rewards_per_episode[best_episode]
    best_action_distribution = action_distribution_per_episode[best_episode]

    print(f"Highest Reward: {best_reward}")
    print(f"Episode: {best_episode}")
    print("Action Distribution Per Hour (Percentages):")
    for hour, actions in best_action_distribution.items():
        print(f"Hour {hour}: {actions}")

    print(agent.validate())
