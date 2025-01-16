import numpy as np
import pandas as pd

from Base.env import DataCenterEnv


environment = DataCenterEnv("../../Dataset/train.xlsx", nr_of_days=1096)

aggregate_reward = 0
terminated = False
state = environment.observation()

while not terminated:
    action = np.random.uniform(-1, 1)
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    #print("Action:", action)
    #print("Next state:", next_state)
    #print("Reward:", reward)

print('Total reward:', aggregate_reward)