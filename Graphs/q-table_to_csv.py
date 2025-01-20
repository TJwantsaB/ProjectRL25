import numpy as np
import pandas as pd
import csv
from env import DataCenterEnv


def run_greedy_and_save(
    path_to_data: str,
    q_table_path: str,
    out_csv: str = "greedy_results.csv",
    max_days: int = None
):
    """
    1) Creates a DataCenterEnv from 'path_to_data'.
    2) Loads a Q-table from 'q_table_path' shaped [storage_bins, price_bins, hour_bins, day_bins, action_size].
       (e.g. 12 x 1 x 24 x 1 x 3).
    3) Runs a single 'greedy' episode from day=1/hour=1 until data or max_days is exhausted.
    4) Saves each step to CSV with columns: [day, hour, price, action, reward].
    5) Prints the total reward at the end.
    
    Returns a list of dicts with {day, hour, price, action, reward}.
    """

    # 1) Create environment
    env = DataCenterEnv(path_to_test_data=path_to_data)
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0

    # 2) Load Q-table
    #    Suppose it's shape [12, 1, 24, 1, 3].
    q_table = np.load(q_table_path)

    # 3) Define your actions in the last dimension => 3 discrete actions
    actions = [-1, 0, 1]   # SELL, NOTHING, BUY  => action_idx in [0..2]

    # 4) A helper function to map (storage, price, hour, day) -> Q-table indices
    #    Since price_bins=1, day_bins=1, we always use index=0 for them.
    #    We'll also clamp the hour to [0..23]. For storage, we do a naive approach:
    #    clamp storage//10 => up to 11. You can refine if needed.
    def get_qtable_indices(obs):
        """
        obs = [storage_level, price, hour, day]
        returns (s_idx, p_idx, h_idx, d_idx)
        """
        storage_level, price, hour, day = obs

        # S) clamp or bin the storage to 12 bins. 
        #    e.g. each bin is 10 MWh if you have 120 MWh total? 
        #    We'll do a naive approach: bin_width=15 => up to 11 for ~160-170 MWh range
        #    Adjust as you see fit. The key is s_idx in [0..11].
        bin_width = 15.0
        s_idx = int(storage_level // bin_width)  # e.g.  0..11
        s_idx = max(0, min(s_idx, 11))          # clamp to [0..11]

        # P) price_bins=1 => always p_idx=0
        p_idx = 0

        # H) hour in [1..24] => h_idx in [0..23]
        h_idx = int(hour) - 1
        if h_idx < 0:
            h_idx = 0
        elif h_idx > 23:
            h_idx = 23

        # D) day_bins=1 => always d_idx=0
        d_idx = 0

        return (s_idx, p_idx, h_idx, d_idx)

    # 5) Step loop
    obs = env.observation()  # => [storage, price, hour, day]
    terminated = False
    results = []  # for collecting day,hour,price,action,reward

    while not terminated:
        # if we only do first N days
        if max_days is not None and env.day > max_days:
            break
        # if out of data
        if env.day >= len(env.price_values):
            break

        # get the Q-table indices for the current state
        (s_idx, p_idx, h_idx, d_idx) = get_qtable_indices(obs)

        # pick the best action from Q-table => shape (12,1,24,1,3)
        best_action_idx = np.argmax(q_table[s_idx, p_idx, h_idx, d_idx, :])
        action_value = actions[best_action_idx]

        # step in environment
        next_obs, rew, terminated_env = env.step(action_value)

        # record
        #   day can be env.day or env.day - 1 if hour just rolled over
        #   We'll just use the day from the OLD obs's perspective:
        day_display = env.day if env.hour != 1 else (env.day - 1)
        if day_display < 1:
            day_display = 1
        
        hour_display = obs[2]  # or env.hour-1
        price_display = obs[1]

        results.append({
            "day": int(day_display),
            "hour": int(hour_display),
            "price": float(price_display),
            "action": float(action_value),
            "reward": float(rew)
        })

        obs = next_obs
        terminated = terminated_env

    # 6) Save results to CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["day","hour","price","action","reward"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Greedy run done. Results saved to {out_csv}.")
    total_r = sum(r["reward"] for r in results)
    print(f"Total reward: {total_r:.2f}")

    return results


if __name__ == "__main__":
    # Example usage
    run_greedy_and_save(
        "../Data/validate.xlsx",
        "final_q_table.npy",
        out_csv="greedy_results_final_validate.csv",
    )
