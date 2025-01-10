import numpy as np
import pandas as pd

from Base.env import DataCenterEnv


def run_baseline(env, number_of_hours, use_fixed_thresholds=False, threshold_1=23.5, threshold_2=84.2):
    aggregate_reward = 0
    terminated = False
    state = env.observation()
    prices = np.empty(0)
    while not terminated:
        if use_fixed_thresholds:
            if len(prices) > 0 and state[1] < threshold_1 and state[0] < 170:
                action = 1
            elif len(prices) > 0 and state[1] > threshold_2:
                action = -1
            else:
                action = 0
        else:
            if len(prices) > 0 and state[1] < np.mean(prices) and state[0] < 170:
                action = 1
            elif state[1] > 200:
                action = -1
            else:
                action = 0

        next_state, reward, terminated = env.step(action)
        prices = np.append(prices, state[1])
        if len(prices) == number_of_hours + 1:
            prices = prices[1:]

        state = next_state
        aggregate_reward += reward
        #print("Action:", action)
        #print("Next state:", next_state)
        #print("Reward:", reward)

    print(f"Total reward({number_of_hours}):", aggregate_reward)

for number_of_hours in range(10, 100):
    environment = DataCenterEnv("../../Dataset/train.xlsx")
    run_baseline(environment, number_of_hours)