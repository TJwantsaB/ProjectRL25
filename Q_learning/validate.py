import argparse
import numpy as np
import sys
from env import DataCenterEnv

ACTIONS = [-1.0, 0.0, 1.0]
# Q-table shape: (12,1,24,1,3) => [storage_bins, price_bins, hour_bins, day_bins, action_dim]
# If you only care about "hour", you'll do something like s_idx=0, p_idx=0, h_idx=(hour-1), d_idx=0.

def get_qtable_indices(observation):
    """
    Because your Q-table has shape (12,1,24,1,3), but you're only using 'hour' as your state,
    we clamp s_idx=0, p_idx=0, d_idx=0, and let h_idx = hour-1.

    obs = [storage_level, price, hour, day]
    hour in 1..24 => h_idx in 0..23
    """
    # Hard-code the other dims to 0 since you have only 1 bin for them
    s_idx = 0  # or whichever index you'd want if you actually used storage
    p_idx = 0
    d_idx = 0

    hour = int(observation[2])  # 1..24
    h_idx = hour - 1
    if h_idx < 0: 
        h_idx = 0
    elif h_idx > 23:
        h_idx = 23

    return (s_idx, p_idx, h_idx, d_idx)

def run_validation(q_table: np.ndarray, val_path: str):
    """
    Runs validation using the given Q-table (shape [12,1,24,1,3]) on val_path.
    Prints total reward at the end.
    """
    env = DataCenterEnv(path_to_test_data=val_path)

    # Manual reset
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    obs = env.observation()  # => [storage_level, price, hour, day]
    done = False
    total_reward = 0.0

    while not done:
        # Convert env.observation -> Q-table indices
        (s_idx, p_idx, h_idx, d_idx) = get_qtable_indices(obs)

        # Pick best action from Q-table last axis
        # q_table[s_idx, p_idx, h_idx, d_idx, :] => shape (3,) => pick argmax
        action_idx = np.argmax(q_table[s_idx, p_idx, h_idx, d_idx, :])
        
        # Convert action_idx -> real action in [-1,0,1]
        action = ACTIONS[action_idx]

        # Step environment
        next_obs, reward, terminated = env.step(action)
        total_reward += reward
        obs = next_obs
        done = terminated

    print("Validation finished.")
    print(f"Total reward over validation data: {total_reward:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qtable",
        type=str,
        default="../Graphs/final_q_table.npy",
        help="Path to the saved Q-table (NumPy .npy file)."
    )
    parser.add_argument(
        "--valdata",
        type=str,
        default="../Data/validate.xlsx",
        help="Path to the Excel file with validation data."
    )
    args = parser.parse_args()

    # Load Q-table
    try:
        q_table = np.load(args.qtable)
    except IOError:
        print(f"Error: Could not load Q-table from {args.qtable}.")
        sys.exit(1)

    print(f"Q-table loaded from {args.qtable} with shape {q_table.shape}.")

    # Validate
    run_validation(q_table, args.valdata)

if __name__ == "__main__":
    main()
