import numpy as np
import random
import gym
from env import DataCenterEnv  # Make sure this import points to your env.py file
import pandas as pd
import wandb


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


class QAgentDataCenter:
    def __init__(
            self,
            environment,
            discount_rate=0.999,
            bin_size_storage=10,
            bin_size_price=1,
            bin_size_trend=1,
            bin_size_hour=24,
            bin_size_day=1,
            bin_size_weekend=1,
            episodes=100,
            learning_rate=0.1,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.95
    ):
        """
        Q-learning agent for the DataCenterEnv.

        The biggest fix we need is to ensure we properly reset the environment
        each episode, because the environment doesn't have a built-in reset() method.
        """
        self.env = environment
        self.discount_rate = discount_rate
        self.bin_size_storage = bin_size_storage
        self.bin_size_price = bin_size_price
        self.bin_size_hour = bin_size_hour
        self.bin_size_day = bin_size_day
        self.bin_size_weekend = bin_size_weekend
        self.bin_size_trend = bin_size_trend

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Define ranges for discretization.
        # You can tune these if you have reason to believe the datacenter might
        # store more or less than 170 MWh, or see higher/lower prices, etc.
        self.storage_min = 0.0
        self.storage_max = 290.0
        self.price_min = 0.01
        self.price_max = 170
        # Hour range is integer 1..24. We'll create 24 bins so each hour is its own bin.
        self.hour_min = 1
        self.hour_max = 24
        # Day range. We can do day modulo 7 or something. We'll do that in `discretize_state`.
        self.day_min = 1
        self.day_max = 365

        # Create bin edges.
        self.bin_storage_edges = np.linspace(
            self.storage_min, self.storage_max, self.bin_size_storage
        )
        # Simple price bins for example
        self.bin_price_edges = np.linspace(
            self.price_min, self.price_max, self.bin_size_price)

        # Trend
        self.bin_trend_edges = [-np.inf, -5, 5, np.inf]

        # Bin edges for hours (1 to 24)
        self.bin_hour_edges = np.linspace(0.5, 24.5, self.bin_size_hour + 1)

        # day of week bins
        self.bin_day_edges = np.linspace(0.5, 7.5, self.bin_size_day + 1)

        # weekend indicator
        self.bin_weekend_edges = np.linspace(0.5, 2.5, self.bin_size_weekend + 1)

        # Discretize the action space. We'll have 5 possible actions in [-1, -0.5, 0, 0.5, 1].
        self.discrete_actions = np.linspace(-1, 1, 3)
        self.action_size = len(self.discrete_actions)

        # Create Q-table: shape = [storage_bins, price_bins, hour_bins, day_bins, action_size]
        self.Q_table = np.zeros(
            (
                self.bin_size_storage,
                self.bin_size_price,
                self.bin_size_hour,
                self.bin_size_day,
                self.bin_size_weekend,
                self.bin_size_trend,
                self.action_size
            )
        )


        # For logging
        self.episode_rewards = []
        self.average_rewards = []

    def discretize_state(self, state_raw, trend):
        """
        Convert continuous state [storage_level, price, hour, day]
        into discrete indices for each dimension.
        """
        storage_level, price, hour, day = state_raw

        # We can do day modulo bin_size_day if we want a repeating pattern:
        day_mod = (day - 1) % self.bin_size_day + 1
        # Calculate weekend indicator
        if self.bin_size_weekend == 2:
            idx_weekend = int(day_mod in [0, 6])  # 1 for Sunday or Saturday, 0 otherwise
        else:
            idx_weekend = 0

        idx_day = np.digitize(day_mod, self.bin_day_edges) - 1
        idx_day = np.clip(idx_day, 0, self.bin_size_day - 1)

        idx_storage = np.digitize(storage_level, self.bin_storage_edges) - 1
        idx_storage = np.clip(idx_storage, 0, self.bin_size_storage - 1)

        # Discretize price, hour, and day
        idx_price = np.digitize(price, self.bin_price_edges) - 1

        idx_hour = np.digitize(hour, self.bin_hour_edges) - 1
        idx_hour = np.clip(idx_hour, 0, self.bin_size_hour - 1)

        # Discretize trend (derivative)
        idx_trend = np.digitize(trend, self.bin_trend_edges) - 1
        idx_trend = np.clip(idx_trend, 0, self.bin_size_trend - 1)

        return (idx_storage, idx_price, idx_hour, idx_day, idx_weekend, idx_trend)

    def epsilon_greedy_action(self, state_disc):
        """
        Pick an action index using epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore with weighted probabilities
            probabilities = [0.16, 0.16, 0.68]  # 10% sell, 20% nothing, 70% buy
            actions = [0, 1, 2]  # 0 -> sell, 1 -> do nothing, 2 -> buy
            return np.random.choice(actions, p=probabilities)
        else:
            # Exploit
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
        wandb.init(project="datacenter-q-learning", name="QAgent_Training", config={
            "episodes": self.episodes,
            "learning_rate": self.learning_rate,
            "discount_rate": self.discount_rate,
            "epsilon_start": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay
        })

        storage_bins = np.linspace(0, 240, 25)  # 25 bins from 0 to 240 (adjust as needed)

        for episode in range(self.episodes):
            print(f"Episode {episode + 1}")

            # Initialize reward logging for the current episode
            reward_logging = {
                "buy": {bin_center: {"reward_sum": 0.0, "count": 0} for bin_center in storage_bins},
                "sell": {bin_center: {"reward_sum": 0.0, "count": 0} for bin_center in storage_bins},
            }

            # Manually reset environment at start of each episode
            state = self.env.reset()
            terminated = False

            total_reward = 0.0

            # # Refit AR(1) model to price values at the start of the episode
            # from statsmodels.tsa.arima.model import ARIMA
            # model = ARIMA(self.env.price_values.flatten(), order=(1, 0, 0))
            # fit = model.fit()
            #
            # # Use the AR model's fitted values (mean predictions)
            # ar_mean = fit.fittedvalues.reshape(self.env.price_values.shape)
            #
            # # Calculate the derivative of the fitted values (trend)
            # ar_trend = np.diff(ar_mean, axis=1, prepend=0)
            ar_trend = np.zeros_like(self.env.price_values)

            while not terminated:
                if self.env.day >= len(self.env.price_values):
                    terminated = True
                    break

                # Discretize state
                state_disc = self.discretize_state(state, ar_trend[int(state[3] - 1), int(state[2] - 1)])

                hours_left = 24 - state[2]
                shortfall = 120 - state[0]
                max_possible_buy = hours_left * 10

                if shortfall > max_possible_buy:
                    action_idx = 1
                    chosen_action = self.discrete_actions[action_idx]
                else:
                    action_idx = self.epsilon_greedy_action(state_disc)
                    chosen_action = self.discrete_actions[action_idx]

                # Step environment
                next_state, reward, terminated = self.env.step(chosen_action)

                def storage_factor_modified_dynamic(storage_level, mid=120.0, alpha=0.02):
                    return -np.tanh(alpha * (mid - storage_level))

                factor = storage_factor_modified_dynamic(state[0])

                if chosen_action > 0:  # Buying action
                    # Smaller reward (lower cost) gives higher shaped reward
                    shaped_reward = -(1000 / max(-reward, 1e-6)) * factor
                    shaped_reward = np.clip(shaped_reward, -20, 20)



                elif chosen_action < 0:  # Selling action
                    # Higher reward (better price) gives higher shaped reward
                    shaped_reward = (reward * factor) / 100


                else:  # No action
                    shaped_reward = reward * factor


                next_state_disc = self.discretize_state(next_state,
                                                        ar_trend[int(next_state[3] - 1), int(next_state[2] - 1)])

                # Q-learning update
                old_value = self.Q_table[
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

                # Calculate TD target
                td_target = shaped_reward + self.discount_rate * next_max
                new_value = old_value + self.learning_rate * (td_target - old_value)
                self.Q_table[
                    state_disc[0],
                    state_disc[1],
                    state_disc[2],
                    state_disc[3],
                    state_disc[4],
                    state_disc[5],
                    action_idx
                ] = new_value

                # Log rewards per storage level and action
                storage_bin = min(storage_bins, key=lambda x: abs(x - state[0]))
                if chosen_action > 0:  # Buying
                    reward_logging["buy"][storage_bin]["reward_sum"] += shaped_reward
                    reward_logging["buy"][storage_bin]["count"] += 1
                elif chosen_action < 0:  # Selling
                    reward_logging["sell"][storage_bin]["reward_sum"] += shaped_reward
                    reward_logging["sell"][storage_bin]["count"] += 1

                total_reward += reward
                state = next_state

            # Decay epsilon after each full episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.episode_rewards.append(total_reward)
            # Log average rewards per storage level for this episode
            for action in ["buy", "sell"]:
                for storage_bin, stats in reward_logging[action].items():
                    if stats["count"] > 0:
                        avg_reward = stats["reward_sum"] / stats["count"]
                        wandb.log({f"{action}_avg_storage_{storage_bin}": avg_reward})

            # Log to wandb
            wandb.log({
                "episode": episode + 1,
                "reward": total_reward,
                "epsilon": self.epsilon
            })

            # Print average every 50 episodes
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.average_rewards.append(avg_reward)
                print(
                    f"Episode {episode + 1}, "
                    f"Avg reward (last 50): {avg_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

            print(total_reward)

        wandb.finish()
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
                state_disc[3],
                state_disc[4],
                state_disc[5]
            ]
        )
        return self.discrete_actions[best_action_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train_clipped_quantiles.xlsx')
    args = parser.parse_args()

    # Create environment
    env = DataCenterEnv(path_to_test_data=args.path)
    # Create agent
    agent = QAgentDataCenter(
        environment=env,
        episodes=100,  # you can reduce or increase
        learning_rate=0.03,
        discount_rate=0.66,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95  # so we see faster decay for demo
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

