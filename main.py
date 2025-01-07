from env import DataCenterEnv
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)

aggregate_reward = 0
terminated = False
state = environment.observation()

while not terminated:
    # agent is your own imported agent class
    # action = agent.act(state)
    action = np.random.uniform(-1, 1)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)