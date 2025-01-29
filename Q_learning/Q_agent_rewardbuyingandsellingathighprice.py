import numpy as np
import random
import csv
import argparse

from env import DataCenterEnv

def rolling_reward(chosen_action, reward, rolling_avg_price):
    """
    Shaped reward now explicitly rewards both buying (+1) and selling (-1).
    This does not differentiate high/low price explicitly,
    but it ensures nonzero actions are positively encouraged beyond the raw reward.

    chosen_action in [-1,0,1]
    reward          = raw_reward from environment
    rolling_avg_price = mean of historical prices so far

    Example logic here:
      - If buy (chosen_action=1), shaped_reward = reward + rolling_avg_price * 10
      - If sell (chosen_action=-1), shaped_reward = reward + rolling_avg_price * 10
      - If do nothing (0), shaped_reward = reward
    """
    shaped_reward = reward
    if chosen_action != 0:
        # Add a positive bonus for any "active" action (buy or sell)
        shaped_reward += rolling_avg_price * 10.0
    return shaped_reward


class QAgentHourOnly:
    def __init__(
        self,
        train_path: str,
        episodes: int = 100,
        discount_rate: float = 0.99,
        learning_rate: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.95,
        max_days: int = None,
        csv_filename: str = "training_stats.csv"
    ):
        """
        A Q-learning agent that uses hour-of-day [1..24] as its state,
        actions = [-1,0,1], and forced logic so the environment doesn't override
        the agent's action unexpectedly. Also uses your rolling_avg reward shaping.

        NOTE:
        - We'll keep a price_history list, appending the current price each step,
          so we can compute the rolling average price.
        """
        self.env = DataCenterEnv(path_to_test_data=train_path)

        self.episodes = episodes
        self.gamma = discount_rate
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_days = max_days
        self.csv_filename = csv_filename

        self.num_states = 24
        # Our 3 discrete actions: sell=-1, none=0, buy=1
        self.actions = [-1, 0, 1]
        self.num_actions = len(self.actions)

        # Q-table shape: [24 states x 3 actions]
        self.Q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        # For forced logic references
        self.daily_energy_demand = 120
        self.max_power_rate = 10

        # For rolling average price
        self.price_history = []

    def _manual_env_reset(self):
        """
        Reset the environment manually, since env.py doesn't provide reset().
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        self.price_history.clear()  # clear rolling price history each episode
        return self.env.observation()

    def _get_state_idx(self, obs):
        """
        obs = [storage_level, price, hour, day], we only care about hour -> [0..23]
        """
        hour = int(obs[2])
        return hour - 1

    def _apply_forced_logic(self, action_value):
        """
        Copy the environment's forced logic checks so the final action used
        is exactly what the environment will do. This avoids the agent
        "thinking" it did a different action than was actually applied.
        """
        storage_level = self.env.storage_level
        hour = self.env.hour
        shortfall = self.daily_energy_demand - storage_level
        hours_left = 24 - hour
        max_possible_buy = hours_left * self.max_power_rate

        final_action = float(np.clip(action_value, -1, 1))

        # (A) If shortfall > max_possible_buy => forcibly buy fraction
        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.max_power_rate)
            if final_action < forced_fraction:
                final_action = forced_fraction

        # (B) Disallow selling if it makes shortfall unfixable
        if final_action < 0:
            sell_mwh = -final_action * self.max_power_rate
            potential_storage = storage_level - sell_mwh
            potential_shortfall = self.daily_energy_demand - potential_storage
            hours_left_after = hours_left - 1
            max_buy_after = hours_left_after * self.max_power_rate

            if potential_shortfall > max_buy_after:
                final_action = 0.0

        return float(np.clip(final_action, -1, 1))

    def epsilon_greedy_action(self, state_idx):
        """
        With probability epsilon: random action in [0..2].
        Otherwise pick argmax from Q_table[state_idx].
        """
        if random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.Q_table[state_idx])

    def train(self):
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
                    # or if out of data
                    if self.env.day >= len(self.env.price_values):
                        terminated = True
                        break

                    # pick a discrete action index
                    action_idx = self.epsilon_greedy_action(state_idx)
                    raw_action_value = self.actions[action_idx]

                    # replicate forced logic
                    corrected_action_value = self._apply_forced_logic(raw_action_value)

                    # Step environment
                    next_obs, raw_reward, terminated_env = self.env.step(corrected_action_value)
                    total_reward += raw_reward

                    # 1) Track the new price for rolling average
                    current_price = obs[1]  # or next_obs[1], up to you
                    self.price_history.append(current_price)
                    rolling_avg_price = np.mean(self.price_history)

                    # 2) Compute shaped reward (our new version that rewards both buy & sell)
                    shaped_reward = rolling_reward(corrected_action_value, raw_reward, rolling_avg_price)

                    # Q-update
                    next_state_idx = self._get_state_idx(next_obs)
                    old_q = self.Q_table[state_idx, action_idx]
                    next_max = np.max(self.Q_table[next_state_idx])

                    td_target = shaped_reward + self.gamma * next_max
                    new_q = old_q + self.alpha * (td_target - old_q)
                    self.Q_table[state_idx, action_idx] = new_q

                    # Move to next step
                    state_idx = next_state_idx
                    obs = next_obs
                    terminated = terminated_env

                # Epsilon decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
                writer.writerow([episode+1, total_reward, self.epsilon])

        print(f"Training finished. Stats saved to {self.csv_filename}.")

    def act(self, obs):
        """
        For greedy usage after training: pick the best action from Q, apply forced logic,
        return final corrected action in [-1..1].
        """
        state_idx = self._get_state_idx(obs)
        best_action_idx = np.argmax(self.Q_table[state_idx])
        raw_action_value = self.actions[best_action_idx]
        corrected = self._apply_forced_logic(raw_action_value)
        return corrected

    def save_q_table(self, filename="q_table.npy"):
        np.save(filename, self.Q_table)
        print(f"Q-table saved to {filename}.")

    def load_q_table(self, filename="q_table.npy"):
        self.Q_table = np.load(filename)
        print(f"Q-table loaded from {filename}.")


def run_validation(agent: QAgentHourOnly, validation_path: str, max_days: int = None):
    """
    Runs a greedy policy on a separate validation dataset,
    also replicating forced logic so final action is consistent.
    Summarizes total reward at the end.
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

        state_idx = agent._get_state_idx(obs)
        best_idx = np.argmax(agent.Q_table[state_idx])
        raw_action_value = agent.actions[best_idx]
        final_action_value = agent._apply_forced_logic(raw_action_value)

        next_obs, rew, terminated_env = val_env.step(final_action_value)
        total_reward += rew
        obs = next_obs
        terminated = terminated_env

    print(f"Validation run finished. Total raw reward: {total_reward:.2f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../Data/train.xlsx')
    parser.add_argument('--val_path', type=str, default='../Data/validate.xlsx')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_days', type=int, default=None)
    parser.add_argument('--val_days', type=int, default=None)
    parser.add_argument('--discount_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_decay', type=float, default=0.95)
    args = parser.parse_args()

    agent = QAgentHourOnly(
        train_path=args.train_path,
        episodes=args.episodes,
        discount_rate=args.discount_rate,
        learning_rate=0.05,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
        max_days=args.max_days,
        csv_filename="../Output/training_stats_rewardbuysell.csv"
    )

    agent.train()
    agent.save_q_table("../Output/q_table_rewardbuysell.npy")

    if args.val_path is not None:
        run_validation(agent, args.val_path, max_days=args.val_days)


if __name__ == "__main__":
    main()
