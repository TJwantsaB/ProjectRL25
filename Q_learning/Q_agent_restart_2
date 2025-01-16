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
        discount_rate: float = 0.99,      # mimic older code that used 0.99
        learning_rate: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.95,      # mimic older code that used 0.95
        max_days: int = None,
        csv_filename: str = "training_stats.csv"
    ):
        """
        A Q-learning agent that ONLY uses hour-of-day [1..24] as its "state".
        Action space in this updated version: [0,1] -> {0="do nothing", 1="buy"}.
        
        We also incorporate:
          - reward shaping: if reward != 0, shaped_reward = 1000 - reward, else 0
          - if storage == 120, we force action=0 for that time step
          - discount_rate=0.99, epsilon_decay=0.95 (like older code)
        
        The environment still has forced logic for shortfalls, etc.
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

        # We only consider hour-of-day as our state: integer in [1..24].
        # We'll store that as an index in [0..23].
        self.num_states = 24

        # In the older code, we used 2 actions: [0,1], i.e. do nothing or buy.
        # self.actions = [0, 1]  # len=2
        self.actions = [-1, 0, 1]  # len=2
        self.num_actions = len(self.actions)

        self.Q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

    def _manual_env_reset(self):
        """
        Because env.py lacks a built-in reset, we manually reset day=1, hour=1, storage=0.
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()  # => [storage, price, hour, day]

    def _get_state_idx(self, obs):
        """
        obs = [storage_level, price, hour, day].
        We'll convert 'hour' to [0..23] index. hour=1 => idx=0, hour=24 => idx=23.
        """
        hour = obs[2]
        return int(hour - 1)

    def epsilon_greedy_action(self, state_idx):
        """
        Epsilon-greedy with 2 actions:
          0 => do nothing
          1 => buy
        """
        if random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.Q_table[state_idx])

    def train(self):
        """
        Main training loop with:
         - reward shaping: if reward != 0 => reward1 = 1000 - reward
         - forced action=0 if storage==120
         - writing [episode, total_reward, epsilon] to CSV
        """
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])

            for episode in range(self.episodes):
                obs = self._manual_env_reset()
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

                    # Epsilon-greedy
                    action_idx = self.epsilon_greedy_action(state_idx)

                    action_value = self.actions[action_idx]  # in {0=do nothing, 1=buy}

                    # Step environment
                    next_obs, raw_reward, terminated_env = self.env.step(action_value)

                    # Reward shaping
                    if raw_reward > 0:
                        shaped_reward = raw_reward -400
                    else:
                        shaped_reward = 0.0

                    total_reward += raw_reward

                    # Q-learning update
                    next_state_idx = self._get_state_idx(next_obs)
                    old_q = self.Q_table[state_idx, action_idx]
                    next_max = np.max(self.Q_table[next_state_idx])

                    td_target = shaped_reward + self.gamma * next_max
                    new_q = old_q + self.alpha * (td_target - old_q)
                    self.Q_table[state_idx, action_idx] = new_q

                    # Move on
                    state_idx = next_state_idx
                    obs = next_obs
                    terminated = terminated or terminated_env

                # Epsilon decay after each episode
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
                writer.writerow([episode+1, total_reward, self.epsilon])

        print(f"Training finished. Stats saved to {self.csv_filename}.")

    def act(self, obs):
        """
        Greedy usage after training. 
        We still only consider hour => state_idx, then do argmax Q.
        Also apply the forced action=0 if storage==120 (like older code).
        """
        state_idx = self._get_state_idx(obs)
        best_action_idx = np.argmax(self.Q_table[state_idx])

        # Force action=0 if storage is 120
        if obs[0] == 120:
            best_action_idx = 0

        return self.actions[best_action_idx]

    def save_q_table(self, filename="q_table.npy"):
        """Optionally save Q-table."""
        np.save(filename, self.Q_table)
        print(f"Q-table saved to {filename}.")

    def load_q_table(self, filename="q_table.npy"):
        """Optionally load Q-table."""
        self.Q_table = np.load(filename)
        print(f"Q-table loaded from {filename}.")


def run_validation(agent: QAgentHourOnly, validation_path: str, max_days: int = None):
    """
    Use the trained Q-table to run on a separate validation dataset (new environment).
    Return the total raw reward (i.e., environment's actual cost or revenue).
    
    We also replicate the older code's forced action=0 if storage=120 
    and do the same shaped reward approach if you want to watch it in action. 
    But for standard evaluation, we just track the raw reward from env.
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

        # Use agent.act, which includes forced action=0 if storage=120
        action = agent.act(obs)
        next_obs, rew, terminated_env = val_env.step(action)
        total_reward += rew
        obs = next_obs
        terminated = terminated or terminated_env

    print(f"Validation run finished. Total raw reward: {total_reward:.2f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../Data/train.xlsx',
                        help="Path to the training dataset (Excel).")
    parser.add_argument('--val_path', type=str, default='../Data/validate.xlsx',
                        help="Path to the validation dataset (Excel). Optional.")
    parser.add_argument('--episodes', type=int, default=100,
                        help="Number of training episodes.")
    parser.add_argument('--max_days', type=int, default=None,
                        help="Only train on the first N days of the dataset.")
    parser.add_argument('--val_days', type=int, default=None,
                        help="If validating, only run for the first N days on the validation dataset.")
    parser.add_argument('--discount_rate', type=float, default=0.8,
                        help="Gamma for Q-learning. Older code used 0.99.")
    parser.add_argument('--epsilon_decay', type=float, default=0.95,
                        help="Epsilon decay. Older code used 0.95.")
    args = parser.parse_args()

    # Create agent for training, using the older code's defaults or your chosen ones
    agent = QAgentHourOnly(
        train_path=args.train_path,
        episodes=args.episodes,
        discount_rate=args.discount_rate,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
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
