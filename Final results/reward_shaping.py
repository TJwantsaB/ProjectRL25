import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

def train_agent(agent_class, agent_kwargs):
    """
    Train a single agent instance.
    :param agent_class: The class of the agent.
    :param agent_kwargs: Dictionary of arguments to initialize the agent.
    :return: rewards_per_episode from agent.train()
    """
    agent = agent_class(**agent_kwargs)
    rewards_per_episode, _ = agent.train()
    return rewards_per_episode

def run_experiments(agent_class, agent_kwargs, n_runs):
    """
    Run multiple instances of the same agent in parallel.
    :param agent_class: The class of the agent.
    :param agent_kwargs: Arguments for initializing the agent.
    :param n_runs: Number of times to run the agent.
    :return: List of rewards_per_episode from all runs.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(train_agent, [(agent_class, agent_kwargs) for _ in range(n_runs)])
    return results


def plot_grouped_convergence(groups, n_episodes, title="Grouped Convergence Plot", save_as_pdf=False,
                             pdf_filename="convergence_plot.pdf"):
    """
    Plot convergence for multiple agent groups with mean and quartile ranges.
    Optionally save the plot as a PDF for editing in tools like Inkscape.

    :param groups: Dictionary where keys are group names and values are lists of rewards_per_episode.
    :param n_episodes: Number of episodes for the plot.
    :param title: Title of the plot.
    :param save_as_pdf: Whether to save the plot as a PDF.
    :param pdf_filename: Name of the PDF file to save the plot.
    """
    plt.figure(figsize=(12, 8))

    for group_name, rewards_list in groups.items():
        # Align rewards data
        all_rewards = np.zeros((len(rewards_list), n_episodes))
        for i, rewards_dict in enumerate(rewards_list):
            for episode, reward in rewards_dict.items():
                all_rewards[i, episode] = reward

        # Compute statistics
        mean_rewards = np.mean(all_rewards, axis=0)
        quartile_25 = np.percentile(all_rewards, 25, axis=0)
        quartile_75 = np.percentile(all_rewards, 75, axis=0)

        # Plot mean and quartile range
        episodes = np.arange(n_episodes)
        plt.plot(episodes, mean_rewards, label=f"{group_name} (Mean)", linewidth=2)
        plt.fill_between(episodes, quartile_25, quartile_75, alpha=0.2, label=f"{group_name} (25-75% Range)")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_as_pdf:
        plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
        print(f"Plot saved as PDF: {pdf_filename}")

    plt.show()

# Example Usage
if __name__ == "__main__":
    from env import DataCenterEnv
    from final_Q import QAgentDataCenter  # Replace with your agent class import

    # Define agent configurations
    agent_a_kwargs = {
        "environment": DataCenterEnv(path_to_test_data="train.xlsx"),
        "episodes": 100,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.95,
        "discount_rate": 0.99,
        "bin_size_price": 1
    }

    agent_b_kwargs = {
        "environment": DataCenterEnv(path_to_test_data="train.xlsx"),
        "episodes": 100,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.95,
        "discount_rate": 0.99,
        "bin_size_price": 1,
        "rolling_window_size": 27
    }

    # Run experiments for Agent A and Agent B
    n_runs = 20  # Number of runs per agent
    agent_a_results = run_experiments(QAgentDataCenter, agent_a_kwargs, n_runs)
    agent_b_results = run_experiments(QAgentDataCenter, agent_b_kwargs, n_runs)

    # Group results for plotting
    groups = {
        "Agent A": agent_a_results,
        "Agent B": agent_b_results,
    }

    # Plot convergence
    plot_grouped_convergence(groups, n_episodes=100, title="Agent Comparison Convergence Plot")
