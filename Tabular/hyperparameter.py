import argparse
import multiprocessing
from env import DataCenterEnv
from mAvg_Storage_outlier import QAgentDataCenter
import os
import wandb


def train_agent(params):
    """
    Train and test the QAgentDataCenter with given parameters.
    Args:
        params (dict): Dictionary containing parameter values for the agent.
    """
    # Create a unique run name based on parameter values
    run_name = (
        f"lr_{params['learning_rate']}_"
        f"dr_{params['discount_rate']}_"
        f"ed_{params['epsilon_decay']}_"
        f"rw_{params['rolling_window_size']}"
    )

    # Initialize W&B for this experiment
    wandb.init(
        project="data_center_outlier_hyper.2",
        config=params,
        name=run_name,  # Set the unique run name
        reinit=True
    )

    # Create environment
    env = DataCenterEnv(path_to_test_data=params['path'])

    # Create agent with given parameters
    agent = QAgentDataCenter(
        environment=env,
        episodes=params['episodes'],
        learning_rate=params['learning_rate'],
        discount_rate=params['discount_rate'],
        epsilon=params['epsilon'],
        epsilon_min=params['epsilon_min'],
        epsilon_decay=params['epsilon_decay'],
        rolling_window_size=params['rolling_window_size'],
        wandb=True  # Pass the W&B flag to the agent
    )

    # Train the agent
    agent.train()

    # Test the agent
    env.day = 1
    env.hour = 1
    env.storage_level = 0.0
    state = env.observation()
    terminated = False
    total_greedy_reward = 0.0

    while not terminated:
        if env.day >= len(env.price_values):
            break
        action = agent.act(state)
        next_state, reward, terminated = env.step(action)
        total_greedy_reward += reward
        state = next_state

    # Log final greedy reward
    wandb.log({"Total Greedy Reward": total_greedy_reward})

    print(f"Parameters: {params}")
    print(f"Total reward using the greedy policy: {total_greedy_reward:.2f}")

    # Finish W&B logging
    wandb.finish()

    # Return results for analysis
    return {
        'params': params,
        'total_reward': total_greedy_reward
    }

def run_multiprocessing_tests(param_grid, processes):
    """
    Run multiple training and testing instances in parallel.
    Args:
        param_grid (list): List of dictionaries with parameter combinations.
        processes (int): Number of processes to run in parallel.
    """
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(train_agent, param_grid)

    # Save results to a file
    output_file = "multiprocessing_results.txt"
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Params: {result['params']}, Total Reward: {result['total_reward']:.2f}\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx', help='Path to the dataset file.')
    parser.add_argument('--processes', type=int, default=10, help='Number of parallel processes.')
    args = parser.parse_args()

    # Define parameter grid for testing
    param_grid = [
        {
            'path': args.path,
            'episodes': 400,
            'learning_rate': lr,
            'discount_rate': dr,
            'epsilon': 1.0,
            'epsilon_min': 0.15,
            'epsilon_decay': ed,
            'rolling_window_size': rw
        }
        for lr in [0.1, 0.01]          # Learning rates to test
        for dr in [0.90, 0.95]                # Discount rates to test
        for ed in [0.97, 0.979]              # Epsilon decay rates to test
        for rw in [3, 6, 9]              # Rolling window sizes to test
    ]

    # Run tests in parallel
    run_multiprocessing_tests(param_grid, processes=args.processes)

