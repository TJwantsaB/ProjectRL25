from Base.env import DataCenterEnv

environment = DataCenterEnv("../../Dataset/train.xlsx")

aggregate_reward = 0
terminated = False
state = environment.observation()

while not terminated:
    if state[2] in [1, 2, 3, 4, 5, 6, 7, 15, 16, 21, 22, 23] or state[1] < 5:
        action = 1
    elif state[1] > 200:
        action = -1
    else:
        action = 0
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    #print("Action:", action)
    #print("Next state:", next_state)
    #print("Reward:", reward)

print('Total reward:', aggregate_reward)