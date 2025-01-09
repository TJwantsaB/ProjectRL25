import numpy as np

from Code.Q_learning.State import State


class QAgent:
    def __init__(self, env, discount_rate=0.99):
        self.env = env
        self.discount_rate = discount_rate
        self.action_space = [-1, 0, 1]

        bin_storage_level = [10 * x for x in range(1, 17)]
        bin_storage_level.append(999999)

        bin_price = [5 * x for x in range(1, 41)]
        bin_price.append(999999)

        bin_hour = [x for x in range(1, 24)]
        bin_hour.append(999999)

        bin_day = [x for x in range(1, 1096)]
        bin_day.append(999999)

        self.bins = [bin_storage_level, bin_price, bin_hour, bin_day]

    def reset_env(self):
        self.env.storage_level = 0
        self.env.hour = 1
        self.env.day = 1
        return self.env.observation()

    def digitize(self, state):
        state.digitize(self.bins)

    def create_Q_table(self):
        bin_lengths = [len(b) for b in self.bins]
        self.Qtable = np.zeros((*bin_lengths, len(self.action_space)))

    def train(self, simulations, learning_rate, epsilon=0.05,
              adaptive_epsilon=False, epsilon_start=1, epsilon_end=0.01, epsilon_decay=1000):
        np.random.seed(0)

        rewards = []
        average_rewards = []

        self.create_Q_table()
        for i in range(simulations):
            done = False
            total_rewards = 0
            obs = self.reset_env()
            state = State(obs[0], obs[1], obs[2], obs[3])
            self.digitize(state)
            if adaptive_epsilon:
                epsilon = np.interp(i, [0, epsilon_decay], [epsilon_start, epsilon_end])

            while not done:
                if np.random.uniform() < epsilon:
                    action_index = np.random.randint(0, len(self.action_space))
                else:
                    action_index = np.argmax(self.Qtable[state.digitized_state[0], state.digitized_state[1],
                                       state.digitized_state[2], state.digitized_state[3], :])

                obs_next_state, reward, done = self.env.step(self.action_space[action_index])
                next_state = State(obs_next_state[0], obs_next_state[1], obs_next_state[2], obs_next_state[3])
                self.digitize(next_state)

                Q_target = (reward + self.discount_rate *
                            np.max(self.Qtable[next_state.digitized_state[0], next_state.digitized_state[1],
                                   next_state.digitized_state[2], next_state.digitized_state[3], :]))

                delta = learning_rate * (reward + Q_target - self.Qtable[state.digitized_state[0], state.digitized_state[1],
                                                                        state.digitized_state[2], state.digitized_state[3], action_index])
                self.Qtable[state.digitized_state[0], state.digitized_state[1],
                            state.digitized_state[2], state.digitized_state[3], action_index] += delta

                total_rewards += reward
                state = next_state

            rewards.append(total_rewards)
            if (i + 1) % 1000 == 0:
                average_rewards.append(sum(rewards) / len(rewards))
                print(f"Average reward after {i + 1} simulations: {average_rewards[-1]} - Epsilon = {epsilon}")
                rewards = []

        print('The simulation is done!')

    def pick_best_action(self, state):
        action_index = np.argmax(self.Qtable[state.digitized_state[0], state.digitized_state[1],
                            state.digitized_state[2], state.digitized_state[3], :])
        return self.action_space[action_index]