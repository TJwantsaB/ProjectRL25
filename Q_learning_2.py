import numpy as np
import random
import gym
from env import DataCenterEnv  # Make sure this import points to your env.py file


class State:
    """
    A small helper class to store (storage_level, price, hour, day) 
    and convert them to discrete bin indices.
    """
    def __init__(self, storage_level, price, hour, day):
        self.storage_level = storage_level
        self.price = price
        self.hour = hour
        self.day = day
        self.digitized_state = (0, 0, 0, 0)

    def digitize(self, bins_storage, bins_price, bins_hour, bins_day):
        """
        Find the discrete bin index for each dimension by comparing value <= bin thresholds.
        """
        s_idx = next(i for i, b in enumerate(bins_storage)  if self.storage_level <= b)
        p_idx = next(i for i, b in enumerate(bins_price)    if self.price         <= b)
        h_idx = next(i for i, b in enumerate(bins_hour)     if self.hour          <= b)
        d_idx = next(i for i, b in enumerate(bins_day)      if self.day           <= b)

        self.digitized_state = (s_idx, p_idx, h_idx, d_idx)


class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=0.99,
        learning_rate=0.1,
        episodes=2000,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    ):
        """
        Q-learning agent for the DataCenterEnv, using bin arrays more like your friend's code.
        """

        self.env = environment
        self.discount_rate = discount_rate  # gamma
        self.learning_rate = learning_rate  # alpha
        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.bins_storage = [10 * x for x in range(1, 17)]
        self.bins_storage.append(999999)

        # self.bins_price = [5 * x for x in range(1, 41)] 
        self.bins_price = self._calculate_price_quantiles(self.env.price_values, 40)
        self.bins_price.append(999999)
        # print(self.bins_price)
        # quit()

        self.bins_hour = [x for x in range(1, 24)]
        self.bins_hour.append(999999)                       

        # day: if you have up to e.g. 365 days, or 3 years = 1095 days, define appropriately:
        self.bins_day = [x for x in range(1, 366)]
        self.bins_day.append(999999)

        self.action_space = [-1, 0, 1]

        # Q-table shape: [len(bins_storage) x len(bins_price) x len(bins_hour) x len(bins_day) x #actions]
        shape = (len(self.bins_storage),
                 len(self.bins_price),
                 len(self.bins_hour),
                 len(self.bins_day),
                 len(self.action_space))
        self.Q_table = np.zeros(shape, dtype=float)

    def _calculate_price_quantiles(self, price_values, num_bins):
        """
        Calculate quantile-based bin edges for prices, focusing only on the middle percentiles.
        This method ensures that price bins are based on data distribution.
        """
        price_values_flat = sorted(price_values.flatten().tolist())
        
        # Compute quantiles for the given number of bins
        quantile_edges = [
            price_values_flat[int(len(price_values_flat) * q)]
            for q in np.linspace(0, 1, num_bins + 1)[1:-1]  # Skip 0% and 100%
        ]
        return quantile_edges

    def _manual_env_reset(self):
        """
        Because env.py has no 'reset' method, we manually reset day, hour, storage_level 
        to start a new episode:
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()

    def epsilon_greedy_action(self, state):
        """
        Epsilon-greedy policy: pick random action with probability epsilon, 
        else pick argmax from Q-table.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore
            return np.random.randint(0, len(self.action_space))
        else:
            # Exploit
            s_idx, p_idx, h_idx, d_idx = state.digitized_state
            return np.argmax(self.Q_table[s_idx, p_idx, h_idx, d_idx, :])

    def train(self):
        for episode in range(self.episodes):
            obs = self._manual_env_reset()
            # Create a State object for digitization
            s = State(obs[0], obs[1], obs[2], obs[3])
            s.digitize(self.bins_storage, self.bins_price, self.bins_hour, self.bins_day)

            terminated = False
            total_reward = 0.0

            while not terminated:
                # If out of data, break:
                if self.env.day >= len(self.env.price_values):
                    terminated = True
                    break

                # Epsilon-greedy
                action_idx = self.epsilon_greedy_action(s)
                action_value = self.action_space[action_idx]

                # Step in environment
                obs_next, reward, terminated = self.env.step(action_value)
                total_reward += reward

                # Next state
                s_next = State(obs_next[0], obs_next[1], obs_next[2], obs_next[3])
                s_next.digitize(self.bins_storage, self.bins_price, self.bins_hour, self.bins_day)

                # Q-learning update
                s_idx, p_idx, h_idx, d_idx = s.digitized_state
                s_next_idx = s_next.digitized_state

                old_val = self.Q_table[s_idx, p_idx, h_idx, d_idx, action_idx]
                best_next_val = np.max(self.Q_table[s_next_idx[0], s_next_idx[1], s_next_idx[2], s_next_idx[3], :])

                # TD target
                td_target = reward + self.discount_rate * best_next_val

                # Update
                self.Q_table[s_idx, p_idx, h_idx, d_idx, action_idx] = (
                    old_val + self.learning_rate * (td_target - old_val)
                )

                # Move to next step
                s = s_next

            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode+1}, Total reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        print("Training finished!")

    def act(self, obs):
        """
        For greedy testing after training.
        """
        s = State(obs[0], obs[1], obs[2], obs[3])
        s.digitize(self.bins_storage, self.bins_price, self.bins_hour, self.bins_day)
        s_idx, p_idx, h_idx, d_idx = s.digitized_state
        best_action_idx = np.argmax(self.Q_table[s_idx, p_idx, h_idx, d_idx, :])
        return self.action_space[best_action_idx]


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
        discount_rate=0.99,
        learning_rate=0.1,
        episodes=50000,    # Increase for better chance of learning
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    )

    agent.train()

    print("\nRunning a quick greedy run after training:")
    # Manual reset
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    obs = env.observation()
    total_greedy_reward = 0.0
    terminated = False

        # Save the Q-table
    np.save("Q_table.npy", agent.Q_table)
    print("Q-table saved as 'Q_table.npy'.")


    while not terminated:
        if env.day >= len(env.price_values):
            break
        action = agent.act(obs)
        next_obs, reward, terminated = env.step(action)
        total_greedy_reward += reward
        obs = next_obs

    print(f"Total reward using the greedy policy after training: {total_greedy_reward:.2f}")
