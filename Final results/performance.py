from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from scipy.stats import levene, shapiro, ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt


def train_and_collect_rewards(agent_class, agent_kwargs):
    """
    Train a single agent and return its best training reward and validation reward.
    """
    agent = agent_class(**agent_kwargs)
    rewards_per_episode, _ = agent.train()
    best_reward = max(rewards_per_episode.values())
    validation_reward = agent.validate()
    return best_reward, validation_reward


def get_best_training_rewards(agent_class, agent_kwargs, n_runs):
    """
    Train multiple agents using multiprocessing and collect their rewards.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(train_and_collect_rewards, [(agent_class, agent_kwargs) for _ in range(n_runs)])

    # Split results into best training rewards and validation rewards
    best_training_rewards = [result[0] for result in results]
    validation_rewards = [result[1] for result in results]

    return best_training_rewards, validation_rewards


def run_agents_parallelly(agent_class, agent_a_kwargs, agent_b_kwargs, n_runs):
    """
    Run Agent A and Agent B experiments concurrently using threads.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(get_best_training_rewards, agent_class, agent_a_kwargs, n_runs)
        future_b = executor.submit(get_best_training_rewards, agent_class, agent_b_kwargs, n_runs)

        # Wait for results
        best_rewards_a, val_rewards_a = future_a.result()
        best_rewards_b, val_rewards_b = future_b.result()

    return (best_rewards_a, val_rewards_a), (best_rewards_b, val_rewards_b)


def compare_rewards(rewards_a, rewards_b, label):
    """
    Perform statistical tests to compare rewards between two agents.
    Handles cases of zero variance or nearly identical data gracefully.
    Prints mean and standard deviation for both agents.
    """
    mean_a = np.mean(rewards_a)
    mean_b = np.mean(rewards_b)
    std_a = np.std(rewards_a)
    std_b = np.std(rewards_b)

    print(f"\nComparison: {label}")
    print(f"  Agent A - Mean: {mean_a:.4f}, Std: {std_a:.4f}")
    print(f"  Agent B - Mean: {mean_b:.4f}, Std: {std_b:.4f}")

    # Check for zero variance
    if np.var(rewards_a) == 0 or np.var(rewards_b) == 0:
        print("  One or both datasets have zero variance. Statistical tests are not meaningful.")
        return {
            "test": "None",
            "statistic": None,
            "p_value": None,
            "p_levene": None,
            "p_shapiro_a": None,
            "p_shapiro_b": None,
        }

    # Check variance with Levene's test
    _, p_levene = levene(rewards_a, rewards_b)

    # Check normality with Shapiro-Wilk test
    _, p_shapiro_a = shapiro(rewards_a)
    _, p_shapiro_b = shapiro(rewards_b)

    if p_shapiro_a > 0.05 and p_shapiro_b > 0.05:
        # If both distributions are normal
        if p_levene > 0.05:
            # Use standard t-test if variances are equal
            test_name = "t-test"
            stat, p_value = ttest_ind(rewards_a, rewards_b, equal_var=True)
        else:
            # Use Welch's t-test if variances are unequal
            test_name = "Welch's t-test"
            stat, p_value = ttest_ind(rewards_a, rewards_b, equal_var=False)
    else:
        # Use Mann-Whitney U test if data is not normal
        test_name = "Mann-Whitney U test"
        stat, p_value = mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')

    # Print test results
    print(f"  Test Used: {test_name}")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Levene's p-value (variance check): {p_levene:.4f}")
    print(f"  Shapiro-Wilk p-value (Agent A): {p_shapiro_a:.4f}")
    print(f"  Shapiro-Wilk p-value (Agent B): {p_shapiro_b:.4f}")

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "p_levene": p_levene,
        "p_shapiro_a": p_shapiro_a,
        "p_shapiro_b": p_shapiro_b,
    }


def plot_boxplot(rewards_a, rewards_b, title, labels):
    """
    Plot a single box plot for rewards (Agent A and Agent B).
    """
    data = [rewards_a, rewards_b]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title(title)
    plt.ylabel("Rewards")
    plt.grid(True)
    plt.show()


def plot_rewards_boxplots(best_rewards_a, best_rewards_b, val_rewards_a, val_rewards_b):
    """
    Plot separate box plots for training and validation rewards.
    """
    plot_boxplot(
        best_rewards_a, best_rewards_b,
        title="Best Training Rewards: Agent A vs Agent B",
        labels=["Agent A", "Agent B"]
    )
    plot_boxplot(
        val_rewards_a, val_rewards_b,
        title="Validation Rewards: Agent A vs Agent B",
        labels=["Agent A", "Agent B"]
    )



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
        "bin_size_price": 5,
        "min_max_price": 50,
    }

    agent_b_kwargs = {
        "environment": DataCenterEnv(path_to_test_data="train.xlsx"),
        "episodes": 20,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.67,
        "discount_rate": 1,
        "discount_min": 0.5,
        "rolling_window_size": 27,
        "storage_factor": 1,
        "bin_size_price": 5,
        "min_max_price": 50,
    }

    # Run experiments for Agent A and Agent B concurrently
    n_runs = 20
    (best_rewards_a, val_rewards_a), (best_rewards_b, val_rewards_b) = run_agents_parallelly(
        QAgentDataCenter, agent_a_kwargs, agent_b_kwargs, n_runs
    )

    # Compare best training rewards
    compare_rewards(best_rewards_a, best_rewards_b, "Best Training Rewards")

    # Compare validation rewards
    compare_rewards(val_rewards_a, val_rewards_b, "Validation Rewards")

    # Plot box plots for rewards
    plot_rewards_boxplots(best_rewards_a, best_rewards_b, val_rewards_a, val_rewards_b)


