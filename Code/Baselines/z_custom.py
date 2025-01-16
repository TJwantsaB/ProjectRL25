import numpy as np

from Base.env import DataCenterEnv


def run_baseline(env):
    aggregate_reward = 0
    terminated = False
    state = env.observation()
    last_price = 0
    while not terminated:
        if state[1] < last_price:
            action = 1
        else:
            action = 0

        last_price = state[1]
        next_state, reward, terminated = env.step(action)

        state = next_state
        aggregate_reward += reward
        #print("Action:", action)
        #print("Next state:", next_state)
        #print("Reward:", reward)

#for number_of_hours in range(10, 100):
environment = DataCenterEnv("../../Dataset/train.xlsx", nr_of_days=1096//3*2)
run_baseline(environment)