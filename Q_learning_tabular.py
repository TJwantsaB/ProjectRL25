import numpy as np
import pandas as pd
import random
import csv
from datetime import datetime
import argparse
import gym

from env import DataCenterEnv 


class State:
    """
    We store and discretize:
        - storage_level
        - price
        - hour
        - day_of_week (1..7)
        - month (1..12)
    """
    def __init__(self, storage_level, price, hour, day_of_week, month):
        self.storage_level = storage_level
        self.price = price
        self.hour = hour
        self.day_of_week = day_of_week
        self.month = month

        # The final discrete state indices
        self.digitized_state = (0, 0, 0, 0, 0)

    def digitize(self, bins_storage, bins_price, bins_hour, bins_day_of_week, bins_month):
        """
        Convert each dimension to an index using the provided bin arrays.
        """
        s_idx = next(i for i, b in enumerate(bins_storage) if self.storage_level <= b)
        p_idx = next(i for i, b in enumerate(bins_price) if self.price <= b)
        h_idx = next(i for i, b in enumerate(bins_hour) if self.hour <= b)
        w_idx = next(i for i, b in enumerate(bins_day_of_week) if self.day_of_week <= b)
        m_idx = next(i for i, b in enumerate(bins_month) if self.month <= b)

        self.digitized_state = (s_idx, p_idx, h_idx, w_idx, m_idx)


class QAgentDataCenter:
    def __init__(
        self,
        environment,
        discount_rate=0.95,
        learning_rate=0.1,
        episodes=1000,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        max_days=None,  # If set, only train on first 'max_days' from the dataset
        csv_filename="training_stats.csv"
    ):
        """
        Q-learning agent that uses day_of_week and month instead of day.

        Args:
          environment: The DataCenterEnv instance (env.py).
          discount_rate: Gamma in Q-learning.
          learning_rate: Alpha in Q-learning.
          episodes: Number of training episodes.
          epsilon, epsilon_min, epsilon_decay: Epsilon-greedy params.
          max_days: If not None, stop each episode if env.day > max_days.
          csv_filename: File to save training stats.
        """
        self.env = environment
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_days = max_days
        self.csv_filename = csv_filename

        # 1) Build a dictionary mapping env.day -> (day_of_week, month).
        self.day2weekmonth = self._build_day_to_weekmonth()

        # 2) Define bins for storage, price, hour, day_of_week, month.
        # Adjust as you wish (finer or coarser).
        self.bins_storage = [10*x for x in range(1, 17)]  # up to 160, then catch-all
        self.bins_storage.append(999999)

        self.bins_price = [5*x for x in range(1, 41)]  # 5..200, catch-all
        self.bins_price.append(999999)

        self.bins_hour = list(range(1, 24))  # hours 1..23
        self.bins_hour.append(999999)

        self.bins_day_of_week = list(range(1, 7))  # 1..6
        self.bins_day_of_week.append(999999)       # 7 in last bin

        self.bins_month = list(range(1, 12))  # 1..11
        self.bins_month.append(999999)        # 12 in last bin

        # 3) Define action space: [-1,0,1] => sell, do nothing, buy
        self.action_space = [-1, 0, 1]

        # 4) Q-table dimension:
        # [storage_bins, price_bins, hour_bins, day_of_week_bins, month_bins, action_size]
        shape = (
            len(self.bins_storage),
            len(self.bins_price),
            len(self.bins_hour),
            len(self.bins_day_of_week),
            len(self.bins_month),
            len(self.action_space)
        )
        self.Q_table = np.zeros(shape, dtype=float)

    def _build_day_to_weekmonth(self):
        """
        Parse the date string from the first column 'PRICES' 
        (like '1/Jan/07') to get day_of_week and month.

        We'll store them in a dict: day -> (day_of_week, month)
        day is 1-based (env.day=1 => index=0).
        """
        day2wm = {}
        date_strings = self.env.test_data["PRICES"].tolist()
        for idx, date_str in enumerate(date_strings):
            # env.day=1 => idx=0
            day_index = idx + 1
            try:
                # parse e.g. '1/Jan/07'
                dt = datetime.strptime(date_str, "%d/%b/%y")
                day_of_week = dt.isoweekday()   # Monday=1, Sunday=7
                month = dt.month               # 1..12
            except ValueError:
                # If format differs, adjust the strptime string or handle error
                raise ValueError(f"Could not parse date string '{date_str}' with %d/%b/%y")

            day2wm[day_index] = (day_of_week, month)
        return day2wm

    def _manual_env_reset(self):
        """
        We manually reset day=1, hour=1, storage=0 
        because env.py doesn't have a built-in reset().
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()

    def epsilon_greedy_action(self, state):
        """
        Epsilon-greedy: random with prob epsilon, else argmax Q.
        """
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, len(self.action_space))
        else:
            # get Q row
            s_idx, p_idx, h_idx, w_idx, m_idx = state.digitized_state
            return np.argmax(self.Q_table[s_idx, p_idx, h_idx, w_idx, m_idx, :])

    def train(self):
        """
        Train the agent for self.episodes episodes.
        For each episode:
          - reset the environment
          - run until terminated or out of days
          - update Q after each step
          - decay epsilon
          - save total_reward + epsilon in CSV
        """
        csv_file = open(self.csv_filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Episode", "TotalReward", "Epsilon"])

        for ep in range(self.episodes):
            obs = self._manual_env_reset()  # => [storage_level, price, hour, day]
            # day_of_week, month from dictionary
            day_of_week, month = self.day2weekmonth[self.env.day]
            s = State(obs[0], obs[1], obs[2], day_of_week, month)
            s.digitize(
                self.bins_storage, self.bins_price, self.bins_hour,
                self.bins_day_of_week, self.bins_month
            )

            terminated = False
            total_reward = 0.0

            while not terminated:
                # If we only want to train on the first N days from the dataset:
                if self.max_days is not None and self.env.day > self.max_days:
                    terminated = True
                    break

                # If env is out of data:
                if self.env.day >= len(self.env.price_values):
                    terminated = True
                    break

                # Epsilon-greedy
                action_idx = self.epsilon_greedy_action(s)
                action_value = self.action_space[action_idx]

                next_obs, reward, terminated_env = self.env.step(action_value)
                total_reward += reward

                # next_obs => [storage_level, price, hour, day]
                # parse day_of_week, month
                day_of_week_next, month_next = self.day2weekmonth.get(self.env.day, (1, 1))
                s_next = State(next_obs[0], next_obs[1], next_obs[2], day_of_week_next, month_next)
                s_next.digitize(
                    self.bins_storage, self.bins_price, self.bins_hour,
                    self.bins_day_of_week, self.bins_month
                )

                # Q-learning update
                (s_idx, p_idx, h_idx, w_idx, m_idx) = s.digitized_state
                (s_next_idx, p_next_idx, h_next_idx, w_next_idx, m_next_idx) = s_next.digitized_state

                old_val = self.Q_table[s_idx, p_idx, h_idx, w_idx, m_idx, action_idx]
                next_max = np.max(
                    self.Q_table[s_next_idx, p_next_idx, h_next_idx, w_next_idx, m_next_idx, :]
                )
                td_target = reward + self.discount_rate * next_max
                new_val = old_val + self.learning_rate * (td_target - old_val)
                self.Q_table[s_idx, p_idx, h_idx, w_idx, m_idx, action_idx] = new_val

                s = s_next
                terminated = terminated or terminated_env

            # end of episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            csv_writer.writerow([ep+1, total_reward, self.epsilon])

        csv_file.close()
        print(f"Training done. Stats saved to {self.csv_filename}.")

    def act(self, obs):
        """
        For using the trained policy greedily.
        """
        # parse day from env.day => day_of_week, month
        day_of_week, month = self.day2weekmonth.get(self.env.day, (1, 1))
        s = State(obs[0], obs[1], obs[2], day_of_week, month)
        s.digitize(
            self.bins_storage, self.bins_price, self.bins_hour,
            self.bins_day_of_week, self.bins_month
        )

        (s_idx, p_idx, h_idx, w_idx, m_idx) = s.digitized_state
        return np.argmax(self.Q_table[s_idx, p_idx, h_idx, w_idx, m_idx, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_days', type=int, default=None,
                        help="Only train on the first N days of the dataset.")
    args = parser.parse_args()

    # 1) Create environment
    env = DataCenterEnv(path_to_test_data=args.path)

    # 2) Create agent
    agent = QAgentDataCenter(
        environment=env,
        discount_rate=0.95,
        learning_rate=0.1,
        episodes=args.episodes,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        max_days=args.max_days,
        csv_filename="training_stats.csv"
    )

    # 3) Train
    agent.train()

    # 4) Quick greedy run
    print("\nGreedy run after training:")
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    obs = env.observation()
    total_greedy = 0.0
    terminated = False

    while not terminated:
        if args.max_days is not None and env.day > args.max_days:
            break
        if env.day >= len(env.price_values):
            break

        action_idx = agent.act(obs)
        action_value = agent.action_space[action_idx]
        next_obs, rew, terminated_env = env.step(action_value)
        total_greedy += rew
        obs = next_obs
        terminated = terminated or terminated_env

    print(f"Total reward (greedy) after training: {total_greedy:.2f}")


if __name__ == "__main__":
    main()
