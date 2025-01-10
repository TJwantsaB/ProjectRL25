import numpy as np

from Base.env import DataCenterEnv
from Code.Q_learning.Attempt_3.QAgent import QAgent
from Code.Q_learning.Attempt_3.State import State


env = DataCenterEnv("../../Dataset/train.xlsx")


simulations = 50000
q_agent = QAgent(env)
q_agent.train(simulations, adaptive_epsilon=True, adaptive_learning_rate=True)

np.save('./Attempt_3/Q_table.npy', q_agent.Qtable)
#q_agent.Qtable = np.load('./Attempt_3/Q_table.npy')


aggregate_reward = 0
terminated = False
obs = q_agent.reset_env()
while not terminated:
    state = State(obs[0], obs[1], obs[2], obs[3])
    q_agent.digitize(state)
    action = q_agent.pick_best_action(state)
    obs_next_state, reward, terminated = q_agent.env.step(action)
    obs = obs_next_state
    aggregate_reward += reward
    #print("Action:", action)
    #print("Next state:", next_state)
    #print("Reward:", reward)

print('FINAL - Total reward:', aggregate_reward)
