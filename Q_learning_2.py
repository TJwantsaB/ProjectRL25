import numpy as np
import random
import gym
from env import DataCenterEnv  # Make sure this import points to your env.py file

class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=0.95,
        bin_size_storage=5,
        bin_size_price=5,
        bin_size_hour=4,
        bin_size_day=7,
        episodes=2000,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    ):
        """
        This class implements a Q-learning agent for the DataCenterEnv environment
        using a tabular approach (similar to the MountainCar Q-learning code).

        Args:
            environment: The DataCenterEnv instance.
            discount_rate: Gamma, discount factor for future rewards.
            bin_size_storage: Number of bins to discretize the storage level.
            bin_size_price: Number of bins to discretize the electricity price.
            bin_size_hour: Number of bins for the hour of day (up to 24).
            bin_size_day: Number of bins for day (you can do day % bin_size_day, for example).
            episodes: Number of episodes (full runs) to train over.
            learning_rate: Alpha, how quickly we update Q-values.
            epsilon: Starting epsilon for epsilon-greedy.
            epsilon_min: Minimum possible epsilon.
            epsilon_decay: Factor to multiply epsilon by after each episode (for gradual decay).
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

        # 1) Define ranges for discretization
        self.storage_min = 0.0
        self.storage_max = 170.0   # A chosen upper bound for storage
        self.price_min = 0.0
        self.price_max = 60.0      # Based on typical price data
        self.hour_min = 1
        self.hour_max = 24
        self.day_min = 1
        self.day_max = 365         # For bigger range or day % 7 approach

        # 2) Create bin edges
        self.bin_storage_edges = np.linspace(
            self.storage_min, self.storage_max, self.bin_size_storage
        )
        self.bin_price_edges = np.linspace(
            self.price_min, self.price_max, self.bin_size_price
        )
        self.bin_hour_edges = np.linspace(
            self.hour_min - 0.5, self.hour_max + 0.5, self.bin_size_hour
        )
        self.bin_day_edges = np.linspace(
            self.day_min - 0.5,
            self.day_min + self.bin_size_day - 0.5,
            self.bin_size_day
        )

        # 3) Discretize the action space
        self.discrete_actions = np.linspace(-1.0, 1.0, num=5)
        self.action_size = len(self.discrete_actions)

        # 4) Create Q-table: shape = [storage_bins, price_bins, hour_bins, day_bins, action_size]
        self.Q_table = np.zeros(
            (
                self.bin_size_storage,
                self.bin_size_price,
                self.bin_size_hour,
                self.bin_size_day,
                self.action_size
            )
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

        # Optionally, mod the day if you only want a repeating cycle:
        day_mod = (day - 1) % self.bin_size_day + 1

        idx_storage = np.digitize(storage_level, self.bin_storage_edges) - 1
        idx_storage = np.clip(idx_storage, 0, self.bin_size_storage - 1)

        idx_price = np.digitize(price, self.bin_price_edges) - 1
        idx_price = np.clip(idx_price, 0, self.bin_size_price - 1)

        idx_hour = np.digitize(hour, self.bin_hour_edges) - 1
        idx_hour = np.clip(idx_hour, 0, self.bin_size_hour - 1)

        idx_day = np.digitize(day_mod, self.bin_day_edges) - 1
        idx_day = np.clip(idx_day, 0, self.bin_size_day - 1)

        return (idx_storage, idx_price, idx_hour, idx_day)

    def epsilon_greedy_action(self, state_disc):
        """
        Pick an action index using epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore
            return np.random.randint(0, self.action_size)
        else:
            # Exploit
            return np.argmax(
                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3]
                ]
            )

    def train(self):
        """
        Train the agent over a number of episodes.
        """
        for episode in range(self.episodes):
            # Reset environment at start of episode
            state = self.env.observation()
            terminated = False

            total_reward = 0.0

            while not terminated:
                # Check if day exceeds the data size
                if self.env.day >= len(self.env.price_values):
                    print(f"Dataset exhausted. Terminating episode {episode + 1}.")
                    terminated = True
                    break

                # Discretize state
                state_disc = self.discretize_state(state)

                # Epsilon-greedy action
                action_idx = self.epsilon_greedy_action(state_disc)
                chosen_action = self.discrete_actions[action_idx]

                # Step environment
                next_state, reward, terminated = self.env.step(chosen_action)

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
                next_max = np.max(
                    self.Q_table[
                        next_state_disc[0],
                        next_state_disc[1],
                        next_state_disc[2],
                        next_state_disc[3]
                    ]
                )
                td_target = reward + self.discount_rate * next_max
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

            # Decay epsilon after each full episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.episode_rewards.append(total_reward)
            # Print average every N episodes
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.average_rewards.append(avg_reward)
                print(f"Episode {episode+1}, Avg reward (last 50): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

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
                state_disc[2],
                state_disc[3]
            ]
        )
        return self.discrete_actions[best_action_idx]


if __name__ == "__main__":
    """
    Example main loop that uses QAgentDataCenter to train on the DataCenterEnv.
    Similar to your main.py, but with tabular Q-learning in place of random actions.
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    # Create environment
    env = DataCenterEnv(path_to_test_data=args.path)

    # Create agent
    agent = QAgentDataCenter(
        environment=env,
        episodes=2000,
        learning_rate=0.1,
        discount_rate=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    )

    # Train
    agent.train()

    # Test run with the greedy policy
    print("\nRunning a quick greedy run with the learned policy:")
    state = env.observation()
    terminated = False
    total_greedy_reward = 0.0

    while not terminated:
        action = agent.act(state)
        next_state, reward, terminated = env.step(action)
        total_greedy_reward += reward
        state = next_state

    print(f"Total reward using the greedy policy after training: {total_greedy_reward:.2f}")
