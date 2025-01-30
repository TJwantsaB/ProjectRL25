import matplotlib.pyplot as plt
import numpy as np

def plot_epsilon_decay(initial_epsilon=1.0, decay_rate=0.99, steps=1000):
    """
    Plots an epsilon decay curve.

    Parameters:
        initial_epsilon (float): The starting value of epsilon.
        decay_rate (float): The multiplicative decay rate (e.g., 0.99 means epsilon = epsilon * decay_rate).
        steps (int): The number of steps over which epsilon decays.
    """
    # Generate steps
    x = np.arange(steps)

    # Compute epsilon values using multiplicative decay
    epsilon = [initial_epsilon]
    for _ in range(1, steps):
        epsilon.append(epsilon[-1] * decay_rate)

    # Plot the curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, epsilon, label="Multiplicative Epsilon Decay Curve")
    plt.xlabel("Steps")
    plt.ylabel("Epsilon")
    plt.title("Multiplicative Epsilon Decay Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
plot_epsilon_decay(initial_epsilon=0.5, decay_rate=0.97, steps=100)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Q-table from the .npy file
q_table = np.load("343.npy")  # Replace with the path to your .npy file

# Squeeze the singleton dimensions
q_table = q_table.squeeze()  # Removes dimensions of size 1

# If the shape is now (12, 24, 3), transpose it to match (24, 12, 3)
if q_table.shape == (15, 24, 3):
    q_table = q_table.transpose(1, 0, 2)  # Transpose to (24, 12, 3)
    print(f"Final shape of the Q-table: {q_table.shape}")

# Print the shape of the loaded data
print(f"Shape of the Q-table: {q_table.shape}")

# Ensure the Q-table has the expected shape
if q_table.shape != (24, 15, 3):
    raise ValueError("The Q-table shape is not 24x12x3. Please check the file.")


# Create a grid for the storage (z-axis) and hour (x-axis)
hours = np.arange(24)  # x-axis
storage = np.arange(15)  # z-axis
actions = range(3)  # Action space indices

# Loop through each action to create a plot
action_names = {0: 'Sell', 1: 'Hold',2 :'Buy'}

for action in actions:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract values for the current action
    values = q_table[:, :, action]

    # Create a meshgrid for plotting
    X, Z = np.meshgrid(hours, storage)
    Y = values.T  # Transpose to align dimensions for plotting

    # Plot the surface
    surf = ax.plot_surface(X, Z, Y, cmap='viridis', edgecolor='none')

    # Add labels and title
    ax.set_xlabel('Hour (1-24)')
    ax.set_ylabel('Storage (0-150)')
    ax.set_zlabel('Q-Value')
    ax.set_title(f'Q-value Gradient Plot for Action: {action_names.get(action, "Unknown")}')

    # Add a color bar to show gradient
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.show()

