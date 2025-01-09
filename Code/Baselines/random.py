import numpy as np
import pandas as pd

from Base.env import DataCenterEnv


data = pd.read_excel("../../Dataset/train.xlsx", nrows=11)
environment = DataCenterEnv(test_data=data)

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