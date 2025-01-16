import numpy as np
import random
import csv
import argparse

from env import DataCenterEnv


class QAgentHourOnly:
    def __init__(
        self,
        train_path: str,
        episodes: int = 100,
        discount_rate: float = 0.95,
        learning_rate: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        max_days: int = None,
        csv_filename: str = "training_stats.csv"
    ):
        """
        A Q-learning agent that only uses the hour-of-day (1..24) as state.
        The action space is [-1, 0, 1].
        
        Args:
          train_path: path to the training Excel file (env uses it).
          episodes: how many training episodes to run.
          discount_rate, learning_rate: standard Q-learning hyperparams.
          epsilon, epsilon_min, epsilon_decay: for epsilon-greedy exploration.
          max_days: if not None, restrict training to first N days of data.
          csv_filename: where to store training stats.
        """
        # Create the environment from the training data
        self.env = DataCenterEnv(path_to_test_data=train_path)

        # Q-learning parameters
        self.episodes = episodes
        self.gamma = discount_rate
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_days = max_days
        self.csv_filename = csv_filename

        # We only consider "hour" as our state -> integer in [1..24].
        # We'll store that as an index in [0..23].
        self.num_states = 24

        # Actions: -1, 0, 1 => we store them in an array for indexing
        self.actions = [-1, 0, 1]  # len=3
        # self.actions = [0, 1]
        self.num_actions = len(self.actions)

        # Create Q-table: shape [24 x 3]
        self.Q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

    def _manual_env_reset(self):
        """
        Because env.py has no built-in reset, we reset day=1, hour=1, storage=0, etc.
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()

    def _get_state_idx(self, obs):
        """
        Convert the environment observation => hour => an integer in [0..23].
        obs = [storage_level, price, hour, day].
        We only care about (hour).
        We'll do: hour_idx = hour-1, so hour=1 => 0, hour=24 => 23.
        """
        hour = obs[2]
        return int(hour - 1)  # ensure 0-based

    def epsilon_greedy_action(self, state_idx):
        """
        Epsilon-greedy: random with probability epsilon, else pick argmax of Q.
        Q_table[state_idx] => array of length 3 => [Q(state, -1), Q(state,0), Q(state,+1)].
        """
        if random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.Q_table[state_idx])

    def train(self):
        """
        Main training loop.
        We write stats (episode, total_reward, epsilon) to CSV.
        """
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])

            for episode in range(self.episodes):
                obs = self._manual_env_reset()  # => [storage, price, hour, day]
                state_idx = self._get_state_idx(obs)
                terminated = False
                total_reward = 0.0

                while not terminated:
                    # If we only want to train on the first N days:
                    if self.max_days is not None and self.env.day > self.max_days:
                        terminated = True
                        break
                    # If environment is out of data:
                    if self.env.day >= len(self.env.price_values):
                        terminated = True
                        break

                    action_idx = self.epsilon_greedy_action(state_idx)
                    action_value = self.actions[action_idx]  # in {-1,0,1}

                    next_obs, reward, terminated_env = self.env.step(action_value)
                    total_reward += reward

                    next_state_idx = self._get_state_idx(next_obs)

                    # Q-learning update:
                    old_q = self.Q_table[state_idx, action_idx]
                    next_max = np.max(self.Q_table[next_state_idx])

                    td_target = reward + self.gamma * next_max
                    new_q = old_q + self.alpha * (td_target - old_q)
                    self.Q_table[state_idx, action_idx] = new_q

                    state_idx = next_state_idx
                    terminated = terminated or terminated_env

                # Epsilon decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
                writer.writerow([episode+1, total_reward, self.epsilon])

        print(f"Training finished. Stats saved to {self.csv_filename}.")

    def act(self, obs):
        """
        For greedy usage after training: pick argmax from Q.
        We still only consider the 'hour' from obs, ignoring the other fields.
        """
        state_idx = self._get_state_idx(obs)
        action_idx = np.argmax(self.Q_table[state_idx])
        return self.actions[action_idx]

    def save_q_table(self, filename="q_table.npy"):
        """
        Save the Q_table to disk for later usage.
        """
        np.save(filename, self.Q_table)
        print(f"Q-table saved to {filename}.")

    def load_q_table(self, filename="q_table.npy"):
        """
        Load Q_table from disk.
        """
        self.Q_table = np.load(filename)
        print(f"Q-table loaded from {filename}.")


def run_validation(agent: QAgentHourOnly, validation_path: str, max_days: int = None):
    """
    Use the trained Q-table to run on a separate validation dataset (new environment).
    Return the total reward.
    """
    print(f"\nRunning validation on {validation_path} with a greedy policy.")
    val_env = DataCenterEnv(path_to_test_data=validation_path)
    val_env.day = 1
    val_env.hour = 1
    val_env.storage_level = 0.0
    obs = val_env.observation()
    total_reward = 0.0
    terminated = False

    while not terminated:
        if max_days is not None and val_env.day > max_days:
            break
        if val_env.day >= len(val_env.price_values):
            break

        action = agent.act(obs)  # agent picks action from Q
        next_obs, rew, terminated_env = val_env.step(action)
        total_reward += rew
        obs = next_obs
        terminated = terminated or terminated_env

    print(f"Validation run finished. Total reward: {total_reward:.2f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../Data/train.xlsx',
                        help="Path to the training dataset (Excel).")
    parser.add_argument('--val_path', type=str, default=None,
                        help="Path to the validation dataset (Excel). Optional.")
    parser.add_argument('--episodes', type=int, default=100,
                        help="Number of training episodes.")
    parser.add_argument('--max_days', type=int, default=None,
                        help="Only train on the first N days of the dataset.")
    parser.add_argument('--val_days', type=int, default=None,
                        help="If validating, only run for the first N days on the validation dataset.")
    args = parser.parse_args()

    # Create agent for training
    agent = QAgentHourOnly(
        train_path=args.train_path,
        episodes=args.episodes,
        discount_rate=0.8,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        max_days=args.max_days,
        csv_filename="../Output/training_stats.csv"
    )

    # Train
    agent.train()

    # Save Q-table if desired
    agent.save_q_table("../Output/q_table.npy")

    # Optionally run a validation dataset if provided
    if args.val_path is not None:
        run_validation(agent, args.val_path, max_days=args.val_days)


if __name__ == "__main__":
    main()
