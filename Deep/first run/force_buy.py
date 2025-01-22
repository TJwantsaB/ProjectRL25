import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gc
import os


def print_memory_usage():
    print("GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        print(
            f"Device {i}: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB allocated, {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB reserved")


def print_device_usage():
    print("Model and Tensor Devices:")
    for name, param in model.named_parameters():
        print(f"{name} on {param.device}")


# Define the TCN backbone
class TCN(nn.Module):
    def __init__(self, input_size, window_size, devices):
        super(TCN, self).__init__()
        self.devices = devices
        # Split layers across devices
        self.causal_conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=2, dilation=1).to(
            devices[0])
        self.conv_blocks_1 = nn.Sequential(
            self._conv_block(32, 32, dilation=1),
            self._conv_block(32, 64, dilation=2),
            self._conv_block(64, 128, dilation=4),
            self._conv_block(128, 128, dilation=8)
        ).to(devices[0])
        self.conv_blocks_2 = nn.Sequential(
            self._conv_block(128, 128, dilation=16),
            self._conv_block(128, 128, dilation=32)
        ).to(devices[1])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1).to(devices[1])

    def _conv_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).to(self.devices[0])  # Switch batch and feature dimensions
        x = self.causal_conv1(x)
        x = self.conv_blocks_1(x)
        x = x.to(self.devices[1])
        x = self.conv_blocks_2(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return x


# Actor-Critic Networks with Target Critic
class ActorCritic(nn.Module):
    class ActorCritic(nn.Module):
        def __init__(self, input_size, window_size, action_space, devices, tau=0.01):
            super(ActorCritic, self).__init__()
            self.tcn = TCN(input_size, window_size, devices)
            self.actor_fc = nn.Linear(128, action_space).to(devices[1])  # Output probabilities over actions
            self.critic_fc = nn.Linear(128, 1).to(devices[1])  # State value
            self.critic_target_fc = nn.Linear(128, 1).to(devices[1])  # Target critic
            self.devices = devices
            self.tau = tau  # Target update factor
            self.temperature = 1.0

            # Initialize target critic to be the same as the current critic
            self.critic_target_fc.load_state_dict(self.critic_fc.state_dict())
            self.temperature = 1.0  # Initialize temperature

        def forward(self, x):
            tcn_out = self.tcn(x)
            # Apply temperature scaling to logits
            logits = self.actor_fc(tcn_out) / self.temperature
            policy = torch.softmax(logits, dim=-1)  # Action probabilities
            value = self.critic_fc(tcn_out)  # State value
            return policy, value

    def update_target_network(self):
        """Soft update of the target critic network."""
        with torch.no_grad():
            for target_param, param in zip(self.critic_target_fc.parameters(), self.critic_fc.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# Apply Action Constraints (forced buy logic)
def apply_action_constraints(action, env):
    """
    Apply constraints to the agent's chosen action based on the environment's rules.
    """
    hours_left = 24 - env.hour
    shortfall = env.daily_energy_demand - env.storage_level
    max_possible_buy = hours_left * env.max_power_rate

    # Force buy if shortfall > max_possible_buy
    if shortfall > max_possible_buy:
        needed_now = shortfall - max_possible_buy
        forced_fraction = min(1.0, needed_now / env.max_power_rate)
        action = max(action, forced_fraction)  # Enforce a minimum buy action

    # Disallow selling if it would make shortfall unfixable
    if action < 0:  # Selling
        sell_mwh = -action * env.max_power_rate
        potential_storage = env.storage_level - sell_mwh
        potential_shortfall = env.daily_energy_demand - potential_storage
        max_buy_after = (hours_left - 1) * env.max_power_rate

        if potential_shortfall > max_buy_after:
            action = 0.0  # Disallow selling

    # Final action within [-1, 1]
    return float(np.clip(action, -1, 1))


# Preprocess observations with price clipping and normalization
def preprocess_observation(obs, max_storage, max_price, min_price=0, max_clip_price=300):
    """
    Normalize and transform a single observation and clip the price between min_price and max_clip_price.
    """
    storage_level, price, hour, day = obs
    storage_level = storage_level / max_storage  # Normalize storage
    price = np.clip(price, min_price, max_clip_price)  # Clip price between 0 and 300
    price = price / max_clip_price  # Normalize price within [0, 1] range after clipping
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return np.array([storage_level, price, hour_sin, hour_cos], dtype=np.float32)


# Normalize rewards
def normalize_rewards(rewards, epsilon=1e-8):
    """
    Normalize rewards by subtracting the mean and dividing by the standard deviation.
    """
    mean = torch.mean(rewards)
    std = torch.std(rewards)
    normalized_rewards = (rewards - mean) / (std + epsilon)  # Prevent division by zero
    return normalized_rewards

#Train Loop
def train_actor_critic(env, model, optimizer, devices, episodes=1000, window_size=168, max_storage=290, max_price=250,
                       gamma=0.95, lr_decay_gamma=0.999):

    # Initialize the learning rate scheduler with exponential decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-5
    )


    for episode in range(episodes):

        model.temperature = max(1.0 * (0.999 ** episode), 0.1)  # Slower temperature decay

        obs = env.reset()
        obs = preprocess_observation(obs, max_storage, max_price)
        sequence_buffer = torch.zeros((window_size, obs.shape[0]), device=devices[0])  # Rolling buffer for observations
        buffer_index = 0  # To manage buffer indexing
        log_probs, values, rewards = [], [], []

        done = False
        total_reward = 0

        while not done:
            # Update the sequence buffer
            sequence_buffer[buffer_index % window_size] = torch.tensor(obs, device=devices[0])
            buffer_index += 1

            # Create the input sequence
            input_sequence = torch.roll(sequence_buffer, -buffer_index,
                                        dims=0) if buffer_index >= window_size else sequence_buffer.clone()

            # Add batch dimension to the input sequence
            input_tensor = input_sequence.unsqueeze(0)

            # Get action and value from the model
            policy, value = model(input_tensor)
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Apply action constraints (forced buy logic)
            action = apply_action_constraints(action.item(), env)

            # Step the environment
            next_obs, reward, done = env.step(action)
            next_obs = preprocess_observation(next_obs, max_storage, max_price)

            # Apply reward shaping
            shaped_reward = reward  # Optionally apply reward shaping here

            # Accumulate logs, values, and rewards
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(shaped_reward)
            total_reward += shaped_reward
            obs = next_obs

        # Normalize and compute discounted returns and advantages
        rewards = torch.tensor(rewards, dtype=torch.float32, device=devices[1])
        normalized_rewards = normalize_rewards(rewards)

        def normalize_returns(returns):
            mean = torch.mean(returns)
            std = torch.std(returns)
            normalized_returns = (returns - mean) / (std + 1e-8)
            return normalized_returns

        returns = torch.zeros_like(normalized_rewards)  # Initialize returns tensor

        G = 0
        for t in reversed(range(len(normalized_rewards))):
            G = normalized_rewards[t] + gamma * G
            returns[t] = G

        # Normalize the computed returns
        returns = normalize_returns(returns)

        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        advantages = returns - values.detach()

        # Calculate training_log.txt
        actor_loss = -(log_probs * advantages).mean()

        # Entropy for exploration
        entropy = dist.entropy().mean()  # Policy entropy
        entropy_weight = 0.005  # Exploration weight
        actor_loss -= entropy_weight * model.temperature * entropy  # Add entropy to encourage exploration

        # Calculate MSE
        critic_loss = torch.nn.MSELoss()(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        # Update model parameters with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Update target network
        model.update_target_network()

        # Apply learning rate decay
        scheduler.step(loss.item())

        # Log metrics every 10 episodes to WANDB
        if (episode + 1) % 20 == 0:
            wandb.log({
                "episode": episode + 1,
                "loss": loss.item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "total_reward": total_reward,
                "entropy": entropy.item(),
                "temperature": model.temperature,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            print(f"[Logged to WANDB] Episode {episode + 1}, Loss: {loss.item():.4f}, Total Reward: {total_reward:.2f}")

        # Cleanup
        del input_tensor, policy, value, dist
        torch.cuda.empty_cache()
        gc.collect()


# Main Execution
if __name__ == "__main__":
    # Load the environment
    from env import DataCenterEnv  # Assuming env.py is in the same directory
    import wandb

    # Initialize Weights & Biases (WANDB)
    wandb.init(
        project="actor_critic_tcn",
        config={
            "episodes": 2000,
            "window_size": 168,
            "input_size": 4,
            "action_space": 3,
            "max_storage": 290,
            "max_price": 250,
            "gamma": 0.95,
            "lr": 1e-2,
            "lr_plateau_factor": 0.9,
            "lr_plateau_patience": 20,
            "tau": 0.01
        }
    )

    path_to_test_data = "../train.xlsx"  # Replace with actual test data path
    env = DataCenterEnv(path_to_test_data, length=365)

    # Hyperparameters
    window_size = 168
    input_size = 4  # Features: normalized storage, normalized price, hour_sin, hour_cos
    action_space = 3  # Example: discrete actions [-1, 0, 1]
    max_storage = 290  # Daily demand + carryover
    max_price = 250  # Estimated max price

    # Device setup
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    # Model and optimizer
    model = ActorCritic(input_size=input_size, window_size=window_size, action_space=action_space, devices=devices)
    # print_device_usage()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_actor_critic(env, model, optimizer, devices, episodes=3000, window_size=window_size, max_storage=max_storage,
                       max_price=max_price)