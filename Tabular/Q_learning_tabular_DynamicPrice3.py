import numpy as np
import random
import gym
from env import DataCenterEnv  # Point to your env.py
import pandas as pd
import argparse
import multiprocessing
import os
import wandb


# ------------------------- Preprocessing Functions --------------------------
def bin_prices(file_path='train.xlsx'):
    # Load the first sheet from the Excel file
    df = pd.read_excel(file_path)  # This will load the first sheet by default

    # Keep the original structure for later reconstruction
    original_columns = df.columns
    original_index = df.index

    # Melt the dataset to consolidate hourly price columns into a single column
    price_data = df.melt(id_vars=['PRICES'],
                         var_name='Hour',
                         value_name='Price')

    # Convert 'PRICES' column to datetime for proper handling
    price_data['PRICES'] = pd.to_datetime(price_data['PRICES'])

    # Sort the data by timestamp for consistency
    price_data = price_data.sort_values(by=['PRICES', 'Hour']).reset_index(drop=True)

    # Add a column for the day of the week
    price_data['Day_of_Week'] = price_data['PRICES'].dt.day_name()

    # Calculate the average price for each day of the week
    weekly_averages = price_data.groupby('Day_of_Week')['Price'].mean()

    # Normalize prices using the weekly average for the corresponding day
    price_data['Weekly_Normalized_Price'] = price_data.apply(
        lambda row: row['Price'] / weekly_averages[row['Day_of_Week']],
        axis=1
    )

    # Shorter rolling window for dynamic percentile-based bins
    rolling_window_size = 48  # Example: last 48 hours
    percentiles = [0, 25, 50, 75, 100]  # Quartiles
    rolling_bins = []

    # Recalculate bins dynamically
    for i in range(len(price_data)):
        if i >= rolling_window_size:
            # Get the rolling window data
            rolling_data = price_data['Weekly_Normalized_Price'].iloc[i - rolling_window_size:i]
            # Compute percentiles for the rolling window
            bins = np.percentile(rolling_data, percentiles)
        else:
            # Default bins when insufficient data
            bins = [0, 0.25, 0.5, 0.75, 1.0]
        rolling_bins.append(bins)

    # Map current prices to dynamically calculated bins
    price_data['Dynamic_Bin_Index'] = [
        max(np.digitize(price_data['Weekly_Normalized_Price'].iloc[i], rolling_bins[i], right=True) - 1, 0)
        for i in range(len(price_data))
    ]

    # Reshape back into the original structure
    reshaped_bins = price_data.pivot(index='PRICES', columns='Hour', values='Dynamic_Bin_Index')

    reshaped_bins = reshaped_bins.copy()

    # Ensure Hour columns are ordered from 1 to 24
    reshaped_bins = reshaped_bins.rename(columns=lambda x: int(str(x).split()[-1]) if isinstance(x, str) else x)

    # Reset the index and set it to range from 1 to the length of the DataFrame
    reshaped_bins.index = range(1, len(reshaped_bins) + 1)

    return reshaped_bins

def bin_prices_with_1day_volatility(file_path='train.xlsx'):
    # Load the first sheet from the Excel file
    df = pd.read_excel(file_path)

    # Melt the dataset to consolidate hourly price columns into a single column
    price_data = df.melt(id_vars=['PRICES'],
                         var_name='Hour',
                         value_name='Price')

    # Clip outliers above 500
    price_data['Price'] = price_data['Price'].clip(upper=500)

    # Convert 'PRICES' column to datetime for proper handling
    price_data['PRICES'] = pd.to_datetime(price_data['PRICES'])

    # Add a column for the day of the week
    price_data['Day_of_Week'] = price_data['PRICES'].dt.day_name()

    # Calculate the average price for each day of the week
    weekly_averages = price_data.groupby('Day_of_Week')['Price'].mean()

    # Normalize prices using the weekly average for the corresponding day
    price_data['Weekly_Normalized_Price'] = price_data.apply(
        lambda row: row['Price'] / weekly_averages[row['Day_of_Week']],
        axis=1
    )

    # Calculate daily volatility and shift to represent the previous day's volatility
    daily_volatility = price_data.groupby(price_data['PRICES'].dt.date)['Weekly_Normalized_Price'].std()
    daily_volatility = daily_volatility.shift(1)

    # Map daily volatility back to the original hourly data
    price_data['Volatility_Past_Day'] = price_data['PRICES'].dt.date.map(daily_volatility)

    price_data = price_data.sort_values(by=['PRICES', 'Hour']).reset_index(drop=True)

    return price_data



# ------------------------- Q-Agent for Immediate Reward ----------------------
class QAgentDataCenterImmediate:
    def __init__(
            self,
            environment,
            preprocessed_prices,
            preprocessed_volatility,
            discount_rate=0.95,
            bin_size_storage=17,
            bin_size_price=5,
            bin_size_volatility=3,
            bin_size_hour=24,
            bin_size_day=7,
            bin_size_month=12,
            episodes=10000,
            learning_rate=0.1,
            epsilon_max=1.0,
            epsilon_min=0.05,
            # For linear decay:
            epsilon_decay_episodes=40000,
            # For exponential decay:
            epsilon_decay_rate=0.9999,
            # For sinusoidal decay:
            sinusoidal_freq=1.0,
            epsilon_decay_strategy='linear'
    ):
        """
        Q-learning agent with immediate reward updates.
        """
        self.env = environment

        # Preprocessing
        self.preprocessed_prices = preprocessed_prices
        self.flattened_prices = self.preprocessed_prices.values.flatten()
        self.preprocessed_volatility = preprocessed_volatility
        self.flattened_volatility = self.preprocessed_volatility['Volatility_Past_Day'].to_numpy()

        # Hyperparams
        self.discount_rate = discount_rate
        self.bin_size_storage = bin_size_storage
        self.bin_size_price = bin_size_price
        self.bin_size_volatility = bin_size_volatility
        self.bin_size_hour = bin_size_hour
        self.bin_size_day = bin_size_day
        self.bin_size_month = bin_size_month

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_strategy = epsilon_decay_strategy
        self.sinusoidal_freq = sinusoidal_freq

        self.epsilon = epsilon_max

        # Define ranges for discretization
        self.storage_min = 0.0
        self.storage_max = 170.0

        # Create bins for storage
        self.bin_storage_edges = np.linspace(self.storage_min, self.storage_max, self.bin_size_storage)

        # Simple price bins for example
        self.bin_price_edges = np.array([0, 1, 2, 3, 4])  # 5 bins

        # Volatility edges
        self.bin_volatility_edges = np.append(
            np.linspace(0, 1, self.bin_size_volatility + 1),
            np.inf
        )

        # Precompute a day-based price/vol index
        self.precomputed_price_volatility = []
        num_days = len(self.flattened_prices)
        for day_idx in range(num_days):
            price_value = self.flattened_prices[day_idx]
            idx_price = np.clip(int(price_value), 0, self.bin_size_price - 1)

            vol_value = self.flattened_volatility[day_idx]
            idx_volatility = np.digitize(vol_value, self.bin_volatility_edges) - 1
            idx_volatility = np.clip(idx_volatility, 0, self.bin_size_volatility - 1)

            self.precomputed_price_volatility.append((idx_price, idx_volatility))

        # Hours directly mapped to bins (1..24)
        self.bin_hour_edges = np.arange(1, 25)

        # Day of week (7 bins)
        self.bin_day_edges = np.arange(1, 8)

        # Month edges (12 bins)
        self.bin_month_edges = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])

        # Discrete actions
        self.discrete_actions = np.linspace(-1.0, 1.0, num=5)
        self.action_size = len(self.discrete_actions)

        # Create Q-table
        self.Q_table = np.zeros(
            (
                self.bin_size_storage,
                self.bin_size_price,
                self.bin_size_volatility + 1,
                self.bin_size_hour,
                self.bin_size_day,
                self.bin_size_month,
                self.action_size
            )
        )

    def discretize_state(self, state_raw):
        """
        state_raw = (storage_level, price, hour, day).
        """
        storage_level, price, hour, day = state_raw

        # Flatten index for day/hour
        flat_idx = int(day * hour)

        # Storage bin
        idx_storage = np.clip(
            np.digitize(storage_level, self.bin_storage_edges) - 1,
            0,
            self.bin_size_storage - 1
        )

        # Price/vol lookup
        if 0 <= flat_idx < len(self.precomputed_price_volatility):
            idx_price, idx_volatility = self.precomputed_price_volatility[flat_idx]
        else:
            day_idx_clamped = np.clip(flat_idx, 0, len(self.precomputed_price_volatility) - 1)
            idx_price, idx_volatility = self.precomputed_price_volatility[day_idx_clamped]

        # Hour bin (0-based index)
        idx_hour = int(hour) - 1

        # Day of week (0..6)
        idx_day = int((day - 1) % 7)

        # Month
        day_in_year = (day - 1) % 365 + 1
        idx_month = np.clip(
            np.digitize(day_in_year, self.bin_month_edges) - 1,
            0,
            self.bin_size_month - 1
        )

        return (
            idx_storage,
            idx_price,
            idx_volatility,
            idx_hour,
            idx_day,
            idx_month
        )

    def epsilon_greedy_action(self, state_disc):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(
                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
                    state_disc[4],
                    state_disc[5]
                ]
            )

    def _manual_env_reset(self):
        """
        Reset environment by manually setting day, hour, storage_level to 0
        or initial values, then return an observation.
        """
        self.env.day = 1
        self.env.hour = 1
        self.env.storage_level = 0.0
        return self.env.observation()

    # ----------------- Epsilon Decay Methods --------------------------------
    def _decay_epsilon_linear(self, episode):
        if episode < self.epsilon_decay_episodes:
            # Linear ramp from epsilon_max to epsilon_min over epsilon_decay_episodes
            self.epsilon = self.epsilon_max - (
                    (self.epsilon_max - self.epsilon_min) *
                    (episode / float(self.epsilon_decay_episodes))
            )
        else:
            self.epsilon = self.epsilon_min

    def _decay_epsilon_exponential(self):
        # Each episode, multiply by epsilon_decay_rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def _decay_epsilon_sinusoidal(self, episode):
        """
        Sinusoidal epsilon decay with cycles from 1 to 0 for a given frequency:
          epsilon(t) = epsilon_min + (epsilon_max - self.epsilon_min) * 0.5 * (1 + cos(pi * freq * episode / total_episodes))
        """
        # Scale frequency to determine cycles from 1 to 0
        scaled_freq = np.pi * self.sinusoidal_freq / self.episodes
        factor = 0.5 * (1 + np.cos(scaled_freq * episode))
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * factor

    def _decay_epsilon(self, episode):
        if self.epsilon_decay_strategy == "linear":
            self._decay_epsilon_linear(episode)
        elif self.epsilon_decay_strategy == "exponential":
            self._decay_epsilon_exponential()
        elif self.epsilon_decay_strategy == "sinusoidal":
            self._decay_epsilon_sinusoidal(episode)
        else:
            raise ValueError(f"Unknown epsilon decay strategy: {self.epsilon_decay_strategy}")

    # ----------------- Training --------------------------------------------
    def train(self):
        print("Training with immediate rewards...")
        for episode in range(self.episodes):
            state = self._manual_env_reset()
            done = False
            total_reward = 0.0

            while not done:
                # If day >= len(price_values), environment is effectively done
                if self.env.day >= len(self.env.price_values):
                    done = True
                    break

                # Discretize state
                state_disc = self.discretize_state(state)

                # Epsilon greedy action
                action_idx = self.epsilon_greedy_action(state_disc)
                chosen_action = self.discrete_actions[action_idx]

                # Take step
                next_state, reward, done = self.env.step(chosen_action)
                total_reward += reward

                # Discretize next state
                next_state_disc = self.discretize_state(next_state)

                # Q-learning update
                old_Q = self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
                    state_disc[4],
                    state_disc[5],
                    action_idx
                ]

                next_max = np.max(
                    self.Q_table[
                        next_state_disc[0],
                        next_state_disc[1],
                        next_state_disc[2],
                        next_state_disc[3],
                        next_state_disc[4],
                        next_state_disc[5]
                    ]
                )

                td_target = reward + self.discount_rate * next_max
                new_Q = old_Q + self.learning_rate * (td_target - old_Q)

                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
                    state_disc[4],
                    state_disc[5],
                    action_idx
                ] = new_Q

                # Move to next state
                state = next_state

            # Decay epsilon
            self._decay_epsilon(episode)

            # Optional logging
            if episode % 100 == 0:
                wandb.log({"episode": episode, "reward": total_reward, "epsilon": self.epsilon})

        print("Finished training!")

    def act(self, state):
        """
        Greedy action selection from trained Q-table.
        """
        state_disc = self.discretize_state(state)
        best_idx = np.argmax(
            self.Q_table[
                state_disc[0],
                state_disc[1],
                state_disc[2],
                state_disc[3],
                state_disc[4],
                state_disc[5]
            ]
        )
        return self.discrete_actions[best_idx]


# -------------------- Multiprocessing Pipeline ------------------------------
def train_and_test_agent(
        path_to_data,
        preprocessed_prices,
        preprocessed_volatility,
        episodes,
        learning_rate,
        discount_rate,
        # Epsilon decays
        epsilon_max,
        epsilon_min,
        epsilon_decay_episodes,  # for linear
        epsilon_decay_rate,  # for exponential
        sinusoidal_freq,  # for sinusoidal
        epsilon_decay_strategy
):
    """
    Each run trains the QAgentDataCenterImmediate with a unique set of parameters
    in a separate process, logs to wandb, tests the greedy policy, and saves Q-table.
    """
    # Make sure each process has a unique random seed
    np.random.seed(os.getpid())
    run_name = f"{epsilon_decay_strategy}_param_{os.getpid()}"

    wandb.init(project="data_center_immediate_reward", reinit=True, name=run_name)
    wandb.config.update({
        "episodes": episodes,
        "learning_rate": learning_rate,
        "discount_rate": discount_rate,
        "epsilon_max": epsilon_max,
        "epsilon_min": epsilon_min,
        "epsilon_decay_episodes": epsilon_decay_episodes,
        "epsilon_decay_rate": epsilon_decay_rate,
        "sinusoidal_freq": sinusoidal_freq,
        "epsilon_decay_strategy": epsilon_decay_strategy
    })

    # Create environment
    env = DataCenterEnv(path_to_test_data=path_to_data)

    # Create agent
    agent = QAgentDataCenterImmediate(
        environment=env,
        preprocessed_prices=preprocessed_prices,
        preprocessed_volatility=preprocessed_volatility,
        discount_rate=discount_rate,
        episodes=episodes,
        learning_rate=learning_rate,
        epsilon_max=epsilon_max,
        epsilon_min=epsilon_min,
        epsilon_decay_episodes=epsilon_decay_episodes,
        epsilon_decay_rate=epsilon_decay_rate,
        sinusoidal_freq=sinusoidal_freq,
        epsilon_decay_strategy=epsilon_decay_strategy
    )

    # Train
    agent.train()

    # Test (greedy)
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

    print(f"Test Reward (decay={epsilon_decay_strategy}, param={os.getpid()}): {total_greedy_reward:.2f}")
    wandb.log({"total_greedy_reward": total_greedy_reward})

    # Save Q-table
    q_table_name = f"Q_table_{epsilon_decay_strategy}_{os.getpid()}.npy"
    np.save(q_table_name, agent.Q_table)

    # Optionally log as artifact
    artifact = wandb.Artifact(f"qtable_{epsilon_decay_strategy}_{os.getpid()}", type="model")
    artifact.add_file(q_table_name)
    wandb.log_artifact(artifact)

    wandb.finish()


# ---------------------------- Main Entry Point -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    args = parser.parse_args()

    # Preprocessing
    preprocessed_prices = bin_prices(args.path)
    preprocessed_volatility = bin_prices_with_1day_volatility(args.path)

    episodes = 50000  # or 50k, adjust as desired
    learning_rate = 0.1
    discount_rate = 0.99
    epsilon_max = 1.0
    epsilon_min = 0.05

    # We define 3 different parameter sets for each strategy:
    # 1) Linear: vary epsilon_decay_episodes
    linear_params = [20000, 40000, 60000]
    # 2) Exponential: vary epsilon_decay_rate
    exponential_params = [0.9999, 0.9995, 0.9990]
    # 3) Sinusoidal: vary freq
    sinusoidal_params = [1.0, 3.0, 5.0]

    tasks = []

    # Linear Decay
    for lin_p in linear_params:
        tasks.append((
            args.path,
            preprocessed_prices,
            preprocessed_volatility,
            episodes,
            learning_rate,
            discount_rate,
            epsilon_max,
            epsilon_min,
            lin_p,  # for linear
            0.9999,  # exponential rate placeholder
            1.0,  # sinusoidal freq placeholder
            "linear"
        ))

    # Exponential Decay
    for exp_p in exponential_params:
        tasks.append((
            args.path,
            preprocessed_prices,
            preprocessed_volatility,
            episodes,
            learning_rate,
            discount_rate,
            epsilon_max,
            epsilon_min,
            40000,  # linear episodes placeholder
            exp_p,
            1.0,  # sinusoidal freq placeholder
            "exponential"
        ))

    # Sinusoidal Decay
    for sin_p in sinusoidal_params:
        tasks.append((
            args.path,
            preprocessed_prices,
            preprocessed_volatility,
            episodes,
            learning_rate,
            discount_rate,
            epsilon_max,
            epsilon_min,
            40000,  # linear episodes placeholder
            0.9999,  # exponential placeholder
            sin_p,
            "sinusoidal"
        ))

    # We have 9 tasks total (3 for each strategy).
    with multiprocessing.Pool(processes=9) as pool:
        pool.starmap(train_and_test_agent, tasks)
