from Base.env import DataCenterEnv

environment = DataCenterEnv("../../Dataset/train.xlsx", nr_of_days=1096)

aggregate_reward = 0
terminated = False
state = environment.observation()

while not terminated:
    if state[2] in [1, 2, 3, 4, 5, 6, 7, 8, 17, 22, 23, 24]:
        action = 1
    else:
        action = 0
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    #print("Action:", action)
    #print("Next state:", next_state)
    #print("Reward:", reward)

print('Total reward:', aggregate_reward)