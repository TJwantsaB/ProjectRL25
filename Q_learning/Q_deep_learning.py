import argparse
import numpy as np
import random
import csv
import collections
import torch
import torch.nn as nn
import torch.optim as optim

from env import DataCenterEnv

# Rolling reward shaping function
def rolling_reward(chosen_action, reward, rolling_avg_price):
    """
    shaped_reward = rolling_avg_price * 10.0 * chosen_action + reward
    """
    return rolling_avg_price * 10.0 * chosen_action + reward

# A small neural network to map 'hour' -> Q-values for [-1,0,1].
# We'll feed hour as a single float, scaled to e.g. 0..23. 
class DQNHourNet(nn.Module):
    def __init__(self, lr=1e-3):
        super(DQNHourNet, self).__init__()
        
        # Our input is just 1 float: "hour_idx" in [0..23].
        # We have 3 outputs => Q-values for actions [-1,0,1].
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # x shape: [batch_size, 1]
        return self.net(x)

# A simple replay buffer
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.memory = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # store transition
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # random sample
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.memory)

# The main agent class for deep Q-learning
class DQAgentHourOnly:
    def __init__(
        self,
        train_path: str,
        episodes: int = 100,
        discount_rate: float = 0.99,
        learning_rate: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.95,
        max_days: int = None,
        csv_filename: str = "training_stats.csv",
        replay_capacity: int = 20000,
        batch_size: int = 32,
        min_replay_size: int = 500
    ):
        """
        A DQN-based agent that uses only 'hour-of-day' as the state (0..23),
        with actions in [-1,0,1].
        Forced logic is replicated to keep consistency with env overrides.
        Rolling average reward shaping is used.

        We'll store transitions in a ReplayBuffer, do mini-batch updates each step.
        """
        # Environment
        self.env = DataCenterEnv(path_to_test_data=train_path)
        self.episodes = episodes
        self.gamma = discount_rate
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_days = max_days
        self.csv_filename = csv_filename

        self.num_states = 24
        self.actions_list = [-1, 0, 1]  # index 0..2
        self.num_actions = len(self.actions_list)

        # forced logic references
        self.daily_energy_demand = 120
        self.max_power_rate = 10

        # rolling average
        self.price_history = []

        # Neural network for Q
        self.q_net = DQNHourNet(lr=self.lr)
        # create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size

    # --------------- HELPER METHODS ---------------

    def _manual_env_reset(self):
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        self.price_history.clear()
        return self.env.observation()  # [storage, price, hour, day]

    def _get_hour_idx(self, obs):
        # hour in obs[2] => int(1..24), shift to 0..23
        return int(obs[2]) - 1

    def _apply_forced_logic(self, action_value):
        """
        replicate environment forced logic to find final_action_value in [-1..1].
        """
        storage_level = self.env.storage_level
        hour = self.env.hour
        shortfall = self.daily_energy_demand - storage_level
        hours_left = 24 - hour
        max_possible_buy = hours_left * self.max_power_rate

        final_action = float(np.clip(action_value, -1, 1))

        # (A) forced buy
        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.max_power_rate)
            if final_action < forced_fraction:
                final_action = forced_fraction

        # (B) disallow selling if it makes shortfall unfixable
        if final_action < 0:
            sell_mwh = -final_action * self.max_power_rate
            potential_storage = storage_level - sell_mwh
            potential_shortfall = self.daily_energy_demand - potential_storage
            hours_left_after = hours_left - 1
            max_buy_after = hours_left_after * self.max_power_rate

            if potential_shortfall > max_buy_after:
                final_action = 0.0

        return float(np.clip(final_action, -1, 1))

    def epsilon_greedy_action(self, hour_idx):
        """
        returns an integer 0..2 => index in self.actions_list
        """
        if random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            # get Q-values from net for that state
            # hour_idx => float tensor shape [1,1], pass into the net
            state_t = torch.tensor([[hour_idx]], dtype=torch.float32)
            with torch.no_grad():
                q_values = self.q_net(state_t)  # shape [1,3]
            best_a_idx = torch.argmax(q_values, dim=1).item()
            return best_a_idx

    def _compute_td_target(self, next_hour_idx, shaped_reward, done):
        """
        If not done, we add gamma*max Q(next), else just shaped_reward
        """
        if done:
            return shaped_reward
        else:
            # next Q
            next_state_t = torch.tensor([[next_hour_idx]], dtype=torch.float32)
            with torch.no_grad():
                next_q_vals = self.q_net(next_state_t)
            next_max = torch.max(next_q_vals).item()
            return shaped_reward + self.gamma * next_max

    # --------------- TRAINING ---------------

    def train(self):
        # We optionally fill some transitions randomly first:
        self._fill_replay_initially()

        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])

            for ep in range(self.episodes):
                obs = self._manual_env_reset()
                hour_idx = self._get_hour_idx(obs)
                done = False
                total_reward = 0.0

                while not done:
                    if self.max_days is not None and self.env.day > self.max_days:
                        done = True
                        break
                    if self.env.day >= len(self.env.price_values):
                        done = True
                        break

                    # 1) Epsilon-greedy
                    action_idx = self.epsilon_greedy_action(hour_idx)
                    raw_action_value = self.actions_list[action_idx]
                    
                    # forced logic
                    final_action_value = self._apply_forced_logic(raw_action_value)

                    # step environment
                    next_obs, raw_reward, terminated = self.env.step(final_action_value)
                    total_reward += raw_reward

                    # rolling price
                    current_price = obs[1]
                    self.price_history.append(current_price)
                    rolling_avg_price = np.mean(self.price_history)

                    # shaped reward
                    shaped_reward = rolling_reward(final_action_value, raw_reward, rolling_avg_price)

                    # next hour
                    next_hour_idx = self._get_hour_idx(next_obs)
                    done_flag = bool(terminated)

                    # store in replay
                    self.replay_buffer.push(hour_idx, action_idx, shaped_reward, next_hour_idx, done_flag)

                    # update net from replay
                    self._train_step()

                    # move on
                    obs = next_obs
                    hour_idx = next_hour_idx
                    done = terminated

                # decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                print(f"Episode {ep+1}, Reward={total_reward:.2f}, Eps={self.epsilon:.3f}")
                writer.writerow([ep+1, total_reward, self.epsilon])

        print("Training finished!")

    def _fill_replay_initially(self):
        """
        Optionally fill the replay with random transitions so we can do minibatch updates from the start.
        """
        # If we want to do a "min_replay_size" approach, do that here
        pass

    def _train_step(self):
        # only train if we have enough samples
        if len(self.replay_buffer) < self.min_replay_size:
            return

        # sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # convert to torch
        states_t = torch.tensor(states, dtype=torch.float32).unsqueeze(-1)  # shape [B,1]
        # actions is shape [B], we gather so we need to shape [B,1]
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # 1) Q(s,a)
        q_values = self.q_net(states_t)  # shape [B,3]
        action_q = q_values.gather(1, actions_t)  # shape [B,1]

        # 2) Q-target
        with torch.no_grad():
            next_q_values = self.q_net(next_states_t)  # shape [B,3]
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]  # shape [B,1]
            targets = rewards_t + (1-dones_t) * self.gamma * max_next_q

        # 3) Loss + optimize
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(action_q, targets)

        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

    # --------------- EVALUATION METHODS ---------------

    def act(self, obs):
        """
        For a *greedy* action after training. No random exploration.
        Also apply forced logic.
        """
        hour_idx = self._get_hour_idx(obs)
        state_t = torch.tensor([[hour_idx]], dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        best_a_idx = torch.argmax(q_vals, dim=1).item()
        raw_action_value = self.actions_list[best_a_idx]
        final_action = self._apply_forced_logic(raw_action_value)
        return final_action

    def save_model(self, fname="dqn_hour.pt"):
        torch.save(self.q_net.state_dict(), fname)
        print(f"Saved model to {fname}")

    def load_model(self, fname="dqn_hour.pt"):
        self.q_net.load_state_dict(torch.load(fname))
        print(f"Loaded model from {fname}")


def run_validation(agent: DQAgentHourOnly, validation_path: str, max_days: int = None):
    """
    We run a greedy policy on validation data, track total reward.
    """
    print(f"\nRunning validation on {validation_path} with a greedy policy.")
    val_env = DataCenterEnv(path_to_test_data=validation_path)
    val_env.day = 1
    val_env.hour = 1
    val_env.storage_level = 0.0

    obs = val_env.observation()
    total_reward = 0.0
    done = False

    while not done:
        if max_days is not None and val_env.day > max_days:
            break
        if val_env.day >= len(val_env.price_values):
            break

        action_value = agent.act(obs)  # => forced logic inside act
        next_obs, rew, terminated = val_env.step(action_value)
        total_reward += rew
        obs = next_obs
        done = terminated

    print(f"Validation finished. Total Reward: {total_reward:.2f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="../Data/train.xlsx")
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_days', type=int, default=None)
    parser.add_argument('--val_days', type=int, default=None)
    parser.add_argument('--discount_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_decay', type=float, default=0.95)
    args = parser.parse_args()

    agent = DQAgentHourOnly(
        train_path=args.train_path,
        episodes=args.episodes,
        discount_rate=args.discount_rate,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
        max_days=args.max_days,
        csv_filename="deepq_training_stats.csv",
        replay_capacity=2000,
        batch_size=32,
        min_replay_size=500
    )

    agent.train()
    agent.save_model("deepq_hour_model.pt")

    if args.val_path is not None:
        run_validation(agent, args.val_path, max_days=args.val_days)


if __name__ == "__main__":
    main()
