################################################################################
# qlearning.py
#
# This file holds the Q-learning tabular solution for our DataCenterEnv
# with a very small state space: we use only the hour of the day (1..24) as state.
# We discretize the action space into 3 discrete actions:
#   0 -> Sell  (maps to -1.0 continuous action)
#   1 -> Hold  (maps to  0.0 continuous action)
#   2 -> Buy   (maps to +1.0 continuous action)
#
# Explanation of each part is given in the comments/docstrings inside the code.
# The structure follows the original MountainCar-v0 Q-learning code closely.
################################################################################

import numpy as np
import random


class QAgentDataCenter():
    """
    QAgentDataCenter implements tabular Q-learning for the simplified DataCenterEnv
    that uses only the hour of the day as the state (i.e. 24 possible states).
    The action space is discretized into 3 actions: Sell (action=0), Hold (action=1), and Buy (action=2).
    """

    def __init__(self, num_hours=24, num_actions=3, discount_rate=0.95):
        """
        Constructor for the tabular Q-learning agent.
        
        Parameters
        ----------
        num_hours : int
            Number of discrete states (hours in a day). We'll treat them as [0..23].
        num_actions : int
            Number of possible discrete actions (here we define 3: Sell, Hold, Buy).
        discount_rate : float
            Gamma parameter for discounting future rewards.
        """

        # Basic settings for the Q-learning agent
        self.num_hours = num_hours
        self.num_actions = num_actions
        self.discount_rate = discount_rate
        
        # Create Q-Table. Shape: [24 x 3], since we only consider hour as our state
        # and we have 3 discrete actions.
        # Initialize to zeros (or small random numbers if you prefer).
        self.Q_table = np.zeros((self.num_hours, self.num_actions))
        
        # We'll fill these in the train(...) method
        self.epsilon = None          # for epsilon-greedy
        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_decay = None
        self.learning_rate = None

    def state_from_env(self, state_obs):
        """
        Convert the environment observation into our "hour-only" discrete state.
        
        The environment's observation is: [storage_level, price, hour, day].
        For now, we ONLY want the 'hour' dimension. 
        We'll return hour-1 so that we can index from 0 to 23 in the Q-table.
        """
        # The hour is in index 2 (third element). The environment labels hours from 1..24.
        # We shift them to 0..23 for indexing in the Q-table.
        hour = int(state_obs[2]) - 1
        return hour

    def choose_action(self, state_discrete):
        """
        Epsilon-greedy action selection.
        
        1) With probability epsilon, choose a random action.
        2) Otherwise, choose the action that maximizes Q(state, action).
        """
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.num_actions)  # random among 0,1,2
        else:
            return np.argmax(self.Q_table[state_discrete, :])

    def train(self,
              env, 
              episodes=10_000,
              learning_rate=0.1,
              epsilon_start=1.0,
              epsilon_end=0.05,
              epsilon_decay=5000):
        """
        Train the Q-learning agent on the given environment.
        
        We do multiple episodes. Each episode ends when the environment is "terminated"
        (i.e., we've gone through all days in the dataset or the env signals done).
        
        Parameters
        ----------
        env : DataCenterEnv
            The environment to train on.
        episodes : int
            Number of full episodes to run. Each episode calls env.reset() and runs until out of data.
        learning_rate : float
            Alpha parameter for the Q-value update rule.
        epsilon_start : float
            Initial epsilon for the epsilon-greedy action selection.
        epsilon_end : float
            Minimum (final) epsilon after decay.
        epsilon_decay : int
            Number of episodes over which epsilon decays linearly from epsilon_start to epsilon_end.
        """
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # For logging average reward, we track reward sums over episodes
        reward_history = []
        
        for episode_i in range(episodes):
            # Linear decay of epsilon over episodes
            # np.interp is a convenient way to linearly interpolate between points.
            self.epsilon = np.interp(
                episode_i,
                [0, self.epsilon_decay],
                [self.epsilon_start, self.epsilon_end]
            )
            
            # Reset environment to start a new episode
            state_obs = env.reset()   # [storage_level, price, hour, day]
            done = False
            total_reward_this_episode = 0.0
            
            while not done:
                # Convert environment's state to our discrete state
                state_discrete = self.state_from_env(state_obs)
                
                # Choose an action with epsilon-greedy
                action_index = self.choose_action(state_discrete)
                
                # Convert the discrete action index to environment's continuous action in [-1,1]
                #   0 -> Sell = -1.0
                #   1 -> Hold =  0.0
                #   2 -> Buy  = +1.0
                continuous_action = None
                if action_index == 0:
                    continuous_action = -1.0
                elif action_index == 1:
                    continuous_action = 0.0
                elif action_index == 2:
                    continuous_action = 1.0
                
                next_state_obs, reward, terminated = env.step(continuous_action)
                
                total_reward_this_episode += reward
                
                # Discretize the next state
                next_state_discrete = self.state_from_env(next_state_obs)
                
                # Q-learning update:
                # Q(s,a) <- Q(s,a) + alpha * ( r + gamma*max(Q(s',:)) - Q(s,a) )
                best_q_next = np.max(self.Q_table[next_state_discrete, :])
                td_target = reward + self.discount_rate * best_q_next
                td_error  = td_target - self.Q_table[state_discrete, action_index]
                
                self.Q_table[state_discrete, action_index] += self.learning_rate * td_error
                
                # Move on to next state
                state_obs = next_state_obs
                done = terminated

            # We finished one episode
            reward_history.append(total_reward_this_episode)
            
            # Optional: print progress every 1000 episodes
            if episode_i % 1000 == 0 and episode_i > 0:
                avg_last_100 = np.mean(reward_history[-100:])
                print(f"Episode {episode_i}, avg reward (last 100 eps): {avg_last_100:.2f}, epsilon: {self.epsilon:.2f}")

            print(f"Episode {episode_i}, total reward: {total_reward_this_episode:.2f}")
        
        print("Training complete. Final Q-table:")
        print(self.Q_table)
        print("Some final stats:")
        print(f"Last 100 episodes average reward: {np.mean(reward_history[-100:]):.2f}")

    def act(self, state_obs):
        """
        This can be used after training to select the best action (greedy) given a state.
        """
        state_discrete = self.state_from_env(state_obs)
        best_action_index = np.argmax(self.Q_table[state_discrete, :])
        
        # Convert best_action_index to continuous action
        if best_action_index == 0:
            return -1.0
        elif best_action_index == 1:
            return 0.0
        else:
            return 1.0


################################################################################
# main.py
#
# Example usage of the QAgentDataCenter class with the DataCenterEnv environment.
# We assume the environment is defined in env.py (as you provided).
# This main script uses the agent to train on the data.
################################################################################

if __name__ == "__main__":
    import argparse
    from env import DataCenterEnv  # make sure env.py is in the same folder
    import numpy as np

    # We'll parse a path to the dataset (Excel file).
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    parser.add_argument('--episodes', type=int, default=5_000,
                        help="Number of episodes to train the Q agent on.")
    args = parser.parse_args()
    
    np.set_printoptions(suppress=True, precision=2)
    path_to_dataset = args.path

    # Create the environment
    env = DataCenterEnv(path_to_dataset)

    # Create the Q-learning agent with hour-only states
    agent = QAgentDataCenter(num_hours=24, num_actions=3, discount_rate=0.95)

    # Train the agent
    agent.train(env,
                episodes=args.episodes,
                learning_rate=0.1,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=2000)

    # After training, we can do a final run to see total reward
    print("\nRunning a final test run with greedy (argmax) actions...")
    state_obs = env.reset()
    done = False
    total_test_reward = 0.0
    while not done:
        # Use agent's best action (greedy)
        continuous_action = agent.act(state_obs)
        next_state_obs, reward, done = env.step(continuous_action)
        total_test_reward += reward
        state_obs = next_state_obs

    print("Total reward in the final test run:", total_test_reward)
