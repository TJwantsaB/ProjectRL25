import argparse
import numpy as np
import random
import csv
import collections
import torch
import torch.nn as nn
import torch.optim as optim

### 1) Reward Shaping ###
def rolling_reward(chosen_action, reward, rolling_avg_price):
    """
    shaped_reward = rolling_avg_price * 10.0 * chosen_action + reward
    """
    return rolling_avg_price * 10.0 * chosen_action + reward


### 2) The neural network for DQN. 
#    We'll feed [price, hour] => (2D input), produce 3 Q-values => for actions [-1,0,1].
class DQNNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_size=128, lr=1e-3):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)  # shape [batch, output_dim]


### 3) Replay Buffer ###
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.memory)


### 4) Our main DQN-based agent class ###
class DQAgent:
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
        min_replay_size: int = 500,
        target_update_freq: int = 2000  # steps after which target network is updated
    ):

        # Env
        self.env = DataCenterEnv(path_to_test_data=train_path)
        self.episodes = episodes
        self.gamma = discount_rate
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_days = max_days
        self.csv_filename = csv_filename

        # Replay
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size

        # Action space = [-1, 0, 1]
        self.actions_list = [-1, 0, 1]

        # Rolling avg price
        self.price_history = []

        # Forced logic references
        self.daily_energy_demand = 120
        self.max_power_rate = 10

        # We define the network with input_dim=2 => [price, hour]
        # output_dim=3 => # of discrete actions
        self.q_net = DQNNet(input_dim=2, output_dim=3, hidden_size=128, lr=self.lr)
        # We'll keep a target net as well
        self.target_net = DQNNet(input_dim=2, output_dim=3, hidden_size=128, lr=self.lr)
        self.target_net.load_state_dict(self.q_net.state_dict())  # init same weights
        self.target_net.eval()

        self.learn_steps = 0
        self.target_update_freq = target_update_freq

    ### HELPER: environment reset
    def _manual_env_reset(self):
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        self.price_history.clear()
        return self.env.observation()

    ### HELPER: state representation => [price, hour]
    #    obs = [storage_level, price, hour, day]
    def _build_state_vec(self, obs):
        price = obs[1]
        hour = obs[2]
        return np.array([price, hour], dtype=np.float32)

    ### Forced logic
    def _apply_forced_logic(self, raw_action):
        # replicate env's forced logic
        # raw_action => float in [-1..1]
        s_level = self.env.storage_level
        hour = self.env.hour
        shortfall = self.daily_energy_demand - s_level
        hours_left = 24 - hour
        max_possible_buy = hours_left * self.max_power_rate

        final_action = float(np.clip(raw_action, -1, 1))

        # (A) forced buy
        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.max_power_rate)
            if final_action < forced_fraction:
                final_action = forced_fraction

        # (B) disallow selling if it makes shortfall unfixable
        if final_action < 0:
            sell_mwh = -final_action * self.max_power_rate
            potential_storage = s_level - sell_mwh
            potential_shortfall = self.daily_energy_demand - potential_storage
            hrs_left_after = hours_left - 1
            max_buy_after = hrs_left_after * self.max_power_rate
            if potential_shortfall > max_buy_after:
                final_action = 0.0

        return float(np.clip(final_action, -1, 1))

    ### Epsilon-greedy (with neural net)
    def select_action(self, state_vec):
        # state_vec => shape [2,] => [price, hour]
        if random.random() < self.epsilon:
            return np.random.randint(0, 3)  # pick 0..2 at random
        else:
            # NN forward
            state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # [1,2]
            with torch.no_grad():
                qvals = self.q_net(state_t)  # shape [1,3]
            best_a_idx = torch.argmax(qvals, dim=1).item()
            return best_a_idx

    ### Train
    def train(self):
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])

            for ep in range(self.episodes):
                obs = self._manual_env_reset()
                done = False
                total_reward = 0.0

                while not done:
                    if self.max_days is not None and self.env.day > self.max_days:
                        done = True
                        break
                    if self.env.day >= len(self.env.price_values):
                        done = True
                        break

                    # state = [price, hour]
                    state_vec = self._build_state_vec(obs)
                    a_idx = self.select_action(state_vec)
                    raw_action_value = self.actions_list[a_idx]

                    # forced logic
                    final_action = self._apply_forced_logic(raw_action_value)

                    next_obs, raw_reward, terminated = self.env.step(final_action)
                    total_reward += raw_reward

                    # rolling price
                    current_price = obs[1]
                    self.price_history.append(current_price)
                    rolling_avg_price = np.mean(self.price_history)

                    # shaped reward
                    shaped_r = rolling_reward(final_action, raw_reward, rolling_avg_price)

                    next_state_vec = self._build_state_vec(next_obs)
                    done_flag = bool(terminated)

                    # Store in replay
                    self.replay_buffer.push(state_vec, a_idx, shaped_r, next_state_vec, done_flag)

                    # Minimally fill replay buffer
                    if len(self.replay_buffer) > self.min_replay_size:
                        self.learn_steps += 1
                        self._train_step()  # sample from replay, do gradient step

                        # Occasionally update target net
                        if self.learn_steps % self.target_update_freq == 0:
                            self.target_net.load_state_dict(self.q_net.state_dict())

                    obs = next_obs
                    done = done_flag

                # Epsilon decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                print(f"Episode {ep+1}, Reward={total_reward:.2f}, Eps={self.epsilon:.3f}")
                writer.writerow([ep+1, total_reward, self.epsilon])

        print("Training finished!")

    ### Single train step from replay
    def _train_step(self):
        # sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # to torch
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)  # shape [B,1]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # forward Q(s)
        qvals = self.q_net(states_t)  # [B,3]
        chosen_q = qvals.gather(1, actions_t)  # [B,1]

        # target Q(s')
        with torch.no_grad():
            # use target net
            qvals_next = self.target_net(next_states_t)
            max_qvals_next = qvals_next.max(dim=1, keepdim=True)[0]  # [B,1]
            targets = rewards_t + (1 - dones_t) * self.gamma * max_qvals_next

        # loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(chosen_q, targets)

        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

    ### Greedy action after training
    def act(self, obs):
        # obs => [storage, price, hour, day]
        state_vec = self._build_state_vec(obs)
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q_net(state_t)
        best_a_idx = torch.argmax(qvals, dim=1).item()
        raw_action = self.actions_list[best_a_idx]
        final_action = self._apply_forced_logic(raw_action)
        return final_action

    def save_model(self, fname="dqn_model.pt"):
        torch.save(self.q_net.state_dict(), fname)
        print(f"Saved model to {fname}")

    def load_model(self, fname="dqn_model.pt"):
        self.q_net.load_state_dict(torch.load(fname))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Loaded model from {fname}")


def run_validation(agent: DQAgent, val_path: str, max_days=None):
    print(f"\nRunning validation on {val_path} with a greedy policy.")
    val_env = DataCenterEnv(path_to_test_data=val_path)
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

        action_value = agent.act(obs)  # forced logic inside
        next_obs, rew, terminated = val_env.step(action_value)
        total_reward += rew
        obs = next_obs
        done = terminated

    print(f"Validation finished. Total Reward: {total_reward:.2f}")
    return total_reward


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/content/drive/MyDrive/Colab Notebooks/Project Reinforcement Learning/train.xlsx')
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max_days', type=int, default=None)
    parser.add_argument('--val_days', type=int, default=None)
    parser.add_argument('--discount_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_decay', type=float, default=0.95)

    # Use parse_known_args to ignore unrecognized arguments
    args, unknown = parser.parse_known_args()

    agent = DQAgent(
        train_path=args.train_path,
        episodes=args.episodes,
        discount_rate=args.discount_rate,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
        max_days=args.max_days,
        csv_filename="/content/drive/MyDrive/Colab Notebooks/Project Reinforcement Learning/deepq_training_stats.csv",
        replay_capacity=30000,
        batch_size=64,
        min_replay_size=1000,
        target_update_freq=2000
    )

    agent.train()
    agent.save_model("dqn_datacenter_model.pt")

    if args.val_path is not None:
        run_validation(agent, args.val_path, max_days=args.val_days)


if __name__ == "__main__":
    main()
