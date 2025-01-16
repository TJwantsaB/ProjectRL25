import argparse
import numpy as np
import sys
from env import DataCenterEnv

ACTIONS = [-1.0, 0.0, 1.0]

def get_hour_state(observation):
    """
    For now this is only for hours. If we want more we have to change this code!
    """
    # observation[2] is hour, an integer 1..24
    hour = int(observation[2])
    state_idx = hour - 1  # shift so 1..24 -> 0..23
    return state_idx


def run_validation(q_table: np.ndarray, val_path: str):
    """
    Runs validation using the given Q-table (numpy array) on the specified dataset path.
    Prints total reward at the end.
    """
    # Create environment for validation
    env = DataCenterEnv(path_to_test_data=val_path)

    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    obs = env.observation()  # [storage_level, price, hour, day]
    done = False
    total_reward = 0.0

    # Run the environment
    while not done:
        # Convert to hour-based discrete state
        state_idx = get_hour_state(obs)
        # Pick the best action from the Q-table (greedy)
        action_idx = np.argmax(q_table[state_idx, :])
        action = ACTIONS[action_idx]

        # Step the environment
        next_obs, reward, terminated = env.step(action)
        total_reward += reward
        obs = next_obs
        done = terminated

        # print(f"Day={env.day}, Hour={env.hour}, Action={action}, Reward={reward}")

    print("Validation finished.")
    print(f"Total reward over validation data: {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qtable",
        type=str,
        default="../Output/q_table.npy",
        help="Path to the saved Q-table (NumPy .npy file)."
    )
    parser.add_argument(
        "--valdata",
        type=str,
        default="../Data/validate.xlsx",
        help="Path to the Excel file with validation data."
    )
    args = parser.parse_args()

    try:
        q_table = np.load(args.qtable)
    except IOError:
        print(f"Error: Could not load Q-table from {args.qtable}.")
        sys.exit(1)

    # Check shape if you like (e.g., should be 24 x 3 if hour-only states)
    print(f"Q-table loaded from {args.qtable} with shape {q_table.shape}.")

    # Run validation
    run_validation(q_table, args.valdata)


if __name__ == "__main__":
    main()

# TEST TEST TEST