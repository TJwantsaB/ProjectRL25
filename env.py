import gym
from gym import spaces
import numpy as np
import pandas as pd


class DataCenterEnv(gym.Env):
    def __init__(self, path_to_test_data):
        super(DataCenterEnv, self).__init__()
        self.continuous_action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.test_data = pd.read_excel(path_to_test_data)
        self.price_values = self.test_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.test_data['PRICES']
        self.start_date = self.test_data['PRICES'].iloc[0]
        self.start_date_day = self.start_date.day_name()


        self.daily_energy_demand = 120  # MWh
        self.max_power_rate = 10  # MW
        self.storage_level = 0
        self.hour = 1
        self.day = 1
        # self.start_date = self.test_data['DATE'][0]

    def step(self, action):
        """
        1. Force buy if you cannot reach 120 by hour 24 with full buying in all remaining hours.
        2. Disallow sell if it makes it impossible to reach 120 with full buying in leftover hours.
        3. Apply final action.
        4. If hour=25, end the day, carry up to +50 above daily demand.
        """
        # Number of hours left in the day (including this one)
        hours_left = 24 - self.hour

        # Current shortfall
        shortfall = self.daily_energy_demand - self.storage_level

        # Max possible buy if we use full power for all remaining hours
        max_possible_buy = hours_left * self.max_power_rate

        # Convert action to [-1,1]
        action = float(np.clip(action, -1, 1))

        # (A) If shortfall > max_possible_buy => forcibly buy extra NOW
        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.max_power_rate)
            # If user action is smaller, override:
            if action < forced_fraction:
                action = forced_fraction

        # (B) Disallow selling if it would make shortfall unfixable
        if action < 0:
            # Proposed sell MWh
            sell_mwh = -action * self.max_power_rate
            # Potential new storage if we allow this sell
            potential_storage = self.storage_level - sell_mwh

            # Potential shortfall after selling
            potential_shortfall = self.daily_energy_demand - potential_storage

            # Hours left AFTER this transaction
            hours_left_after = hours_left - 1  # because we’re using up this hour
            max_buy_after = hours_left_after * self.max_power_rate

            # If potential_shortfall > max_buy_after => can’t fix it => disallow sell
            if potential_shortfall > max_buy_after:
                action = 0.0  # do nothing instead of selling

        # Final action in [-1,1]
        action = float(np.clip(action, -1, 1))

        energy_transacted = action * self.max_power_rate

        # Apply transaction
        price = self.price_values[self.day - 1][self.hour - 1]

        if energy_transacted > 0:
            # Buying
            buy_amount = energy_transacted # min(energy_transacted, self.max_storage_capacity - self.storage_level)
            cost = buy_amount * price
            self.storage_level += buy_amount
            reward = -cost
        else:
            # Selling
            sell_amount = min(-energy_transacted, self.storage_level)
            revenue = sell_amount * price * 0.8
            self.storage_level -= sell_amount
            reward = revenue

        # Next hour
        self.hour += 1

        # If the day is over (hour=25):
        if self.hour == 25:
            self.day += 1
            self.hour = 1
            # End-of-day logic: keep up to +50 MWh over daily requirement
            surplus = max(self.storage_level - self.daily_energy_demand, 0.0)
            carryover = min(surplus, 50.0)
            self.storage_level = 0.0 + carryover  # reset day, keep carryover

        # Terminate if out of data
        terminated = (self.day >= len(self.price_values))

        return self.observation(), reward, terminated

    def observation(self):
        price = self.price_values[self.day - 1][self.hour - 1]
        self.state = np.array([self.storage_level, price, self.hour, self.day])
        return self.state

    def reset(self):
        self.hour = 1
        self.day = 1
        self.storage_level = 0
        return self.observation()
