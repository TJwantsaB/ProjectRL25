import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro, ttest_ind, mannwhitneyu
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def calculate_hourly_entropy(action_distribution):
    """
    Calculate the entropy for each hour in a given action distribution.
    """
    hourly_entropy = {}
    for hour, actions in action_distribution.items():
        probabilities = [p / 100 for p in actions.values()]  # Convert percentages to probabilities
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        hourly_entropy[hour] = entropy
    return hourly_entropy


def calculate_best_episode_hourly_entropy(action_distribution_per_episode, rewards_per_episode):
    """
    Calculate the hourly entropy for the best episode (highest reward).
    """
    # Find the best episode (highest reward)
    best_episode = max(rewards_per_episode, key=rewards_per_episode.get)
    best_action_distribution = action_distribution_per_episode[best_episode]
    return calculate_hourly_entropy(best_action_distribution)


def train_agent(agent_class, agent_kwargs):
    """
    Train a single agent instance and return results for analysis.
    """
    agent = agent_class(**agent_kwargs)
    rewards_per_episode, action_distribution_per_episode = agent.train()
    hourly_entropy = calculate_best_episode_hourly_entropy(action_distribution_per_episode, rewards_per_episode)
    return hourly_entropy


def run_agents_concurrently(agent_class, agent_kwargs, n_runs):
    """
    Train multiple agents concurrently using multiprocessing.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(train_agent, [(agent_class, agent_kwargs) for _ in range(n_runs)])
    return results


def run_agents_parallelly(agent_a_kwargs, agent_b_kwargs, agent_class, n_runs):
    """
    Run Agent A and Agent B experiments concurrently using threading.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_agents_concurrently, agent_class, agent_a_kwargs, n_runs)
        future_b = executor.submit(run_agents_concurrently, agent_class, agent_b_kwargs, n_runs)

        hourly_entropies_a = future_a.result()
        hourly_entropies_b = future_b.result()

    return hourly_entropies_a, hourly_entropies_b


def plot_hourly_entropy_comparison(hourly_entropies_a, hourly_entropies_b, title="Hourly Entropy Comparison"):
    """
    Plot mean hourly entropy with error bars for two agents.
    """
    hours = range(1, 25)

    # Handle missing hours and calculate statistics for Agent A
    entropies_a = [
        [hourly_entropies.get(hour, 0) for hour in hours]  # Default to 0 if hour is missing
        for hourly_entropies in hourly_entropies_a
    ]
    mean_entropy_a = np.mean(entropies_a, axis=0)
    std_entropy_a = np.std(entropies_a, axis=0)

    # Handle missing hours and calculate statistics for Agent B
    entropies_b = [
        [hourly_entropies.get(hour, 0) for hour in hours]  # Default to 0 if hour is missing
        for hourly_entropies in hourly_entropies_b
    ]
    mean_entropy_b = np.mean(entropies_b, axis=0)
    std_entropy_b = np.std(entropies_b, axis=0)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.errorbar(hours, mean_entropy_a, yerr=std_entropy_a, label="Agent A", fmt='-o', capsize=5)
    plt.errorbar(hours, mean_entropy_b, yerr=std_entropy_b, label="Agent B", fmt='-o', capsize=5)
    plt.xlabel("Hour")
    plt.ylabel("Entropy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def test_overall_entropy(hourly_entropies_a, hourly_entropies_b):
    """
    Perform a single test for overall entropy differences across all hours between two agents.
    Automatically handles variance checks, normality checks, and selects the appropriate test.

    Parameters:
    - hourly_entropies_a: List of entropy dictionaries (one per run) for Agent A
    - hourly_entropies_b: List of entropy dictionaries (one per run) for Agent B

    Returns:
    - A dictionary with the test result (test used, statistic, and p-value).
    """
    # Flatten the entropy values across all hours and all runs
    entropies_a = [hourly_entropies.get(hour, 0) for hourly_entropies in hourly_entropies_a for hour in range(1, 25)]
    entropies_b = [hourly_entropies.get(hour, 0) for hourly_entropies in hourly_entropies_b for hour in range(1, 25)]

    # Check variance with Levene's test
    _, p_levene = levene(entropies_a, entropies_b)

    # Check normality with Shapiro-Wilk test
    _, p_shapiro_a = shapiro(entropies_a)
    _, p_shapiro_b = shapiro(entropies_b)

    if p_shapiro_a > 0.05 and p_shapiro_b > 0.05:
        # If both distributions are normal
        if p_levene > 0.05:
            # Use standard t-test if variances are equal
            test_name = "t-test"
            stat, p_value = ttest_ind(entropies_a, entropies_b, equal_var=True)
        else:
            # Use Welch's t-test if variances are unequal
            test_name = "Welch's t-test"
            stat, p_value = ttest_ind(entropies_a, entropies_b, equal_var=False)
    else:
        # Use Mann-Whitney U test if data is not normal
        test_name = "Mann-Whitney U test"
        stat, p_value = mannwhitneyu(entropies_a, entropies_b, alternative='two-sided')

    # Return the test result
    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "p_levene": p_levene,
        "p_shapiro_a": p_shapiro_a,
        "p_shapiro_b": p_shapiro_b,
    }


if __name__ == "__main__":
    from env import DataCenterEnv
    from final_Q import QAgentDataCenter  # Replace with your agent class import

    # Define agent configurations
    agent_a_kwargs = {
        "environment": DataCenterEnv(path_to_test_data="train.xlsx"),
        "episodes": 20,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.67,
        "discount_rate": 1,
        "rolling_window_size": 27,
        "storage_factor": 1,
    }

    agent_b_kwargs = {
        "environment": DataCenterEnv(path_to_test_data="train.xlsx"),
        "episodes": 20,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.67,
        "discount_rate": 1,
        "rolling_window_size": 27,
        "storage_factor": 1,
        "bin_size_price": 5,
        "min_max_price": 50,
    }

    # Run experiments for Agent A and Agent B concurrently
    n_runs = 10
    hourly_entropies_a, hourly_entropies_b = run_agents_parallelly(agent_a_kwargs, agent_b_kwargs, QAgentDataCenter,
                                                                   n_runs)

    # Plot comparison of entropy
    plot_hourly_entropy_comparison(hourly_entropies_a, hourly_entropies_b, "Entropy Comparison: Agent A vs Agent B")

    # Test overall entropy difference
    result = test_overall_entropy(hourly_entropies_a, hourly_entropies_b)

    print(f"Test Used: {result['test']}")
    print(f"Statistic: {result['statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Levene's p-value (variance check): {result['p_levene']:.4f}")
    print(f"Shapiro-Wilk p-value (Agent A): {result['p_shapiro_a']:.4f}")
    print(f"Shapiro-Wilk p-value (Agent B): {result['p_shapiro_b']:.4f}")
