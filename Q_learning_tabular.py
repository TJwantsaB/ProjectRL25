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
        bin_size_hour=12,     
        bin_size_day=7,        
        episodes=2000,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    ):
        """
        Q-learning agent for the DataCenterEnv with price bins based on quantiles.
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

        # Define ranges for discretization.
        self.storage_min = 0.0
        self.storage_max = 170.0
        self.price_min = 0.01
        self.price_max = 2500.0
        self.hour_min = 1
        self.hour_max = 24
        self.day_min = 1
        self.day_max = 365

        # Create quantile-based bins for price
        self.bin_storage_edges = np.linspace(
            self.storage_min, self.storage_max, self.bin_size_storage
        )
        
        self.bin_price_edges = self._calculate_price_quantiles(self.env.price_values, self.bin_size_price)
        # print(self.bin_price_edges)
        # quit()
        
        self.bin_hour_edges = np.linspace(
            self.hour_min - 0.5, self.hour_max + 0.5, self.bin_size_hour
        )
        self.bin_day_edges = np.linspace(
            self.day_min - 0.5,
            self.day_min + self.bin_size_day - 0.5,
            self.bin_size_day
        )

        # Discretize the action space. We'll have 5 possible actions in [-1, -0.5, 0, 0.5, 1].
        self.discrete_actions = np.linspace(-1.0, 1.0, num=5)
        self.action_size = len(self.discrete_actions)

        # Create Q-table
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

    def _calculate_price_quantiles(self, price_values, num_bins):
        """
        Calculate quantile-based bin edges for prices, focusing only on the middle percentiles.
        The lowest and highest values in the data will automatically fall into the closest bins.
        """
        price_values_flat = np.array(price_values).flatten()
        
        # Calculate quantiles, evenly dividing the data into num_bins sections.
        quantile_edges = np.quantile(price_values_flat, np.linspace(0, 1, num_bins + 1)[1:-1])  # Skip 0% and 100%
        
        return quantile_edges


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
                # If day >= len(price_values), environment says it's out of data
                # but let's break gracefully if that happens for the *current* episode.
                # We'll then do a fresh reset next episode.

                #  TODO: Check if next state the last state! 
                if self.env.day >= len(self.env.price_values):
                    # This is where the environment is done for the data set
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

            print(f"Total reward: {total_reward}")

            # chosen_action = self.discrete_actions[action_idx]
            # print(f"Day {self.env.day}, Hour {self.env.hour}, Action: {chosen_action}")
            # next_state, reward, terminated = self.env.step(chosen_action)
            # print(f"Reward: {reward}, Next state: {next_state}")

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    # Create environment
    env = DataCenterEnv(path_to_test_data=args.path)

    # Create agent
    agent = QAgentDataCenter(
        environment=env,
        episodes=100,         # you can reduce or increase
        learning_rate=0.1,
        discount_rate=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95   # so we see faster decay for demo
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
