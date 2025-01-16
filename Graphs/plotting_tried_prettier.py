import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def plot_prices_and_actions(
    csv_file: str,
    out_png: str = "prices_actions_plot.png",
    day_start: int = 10,
    day_end: int = 20,
    vmin: float = -2.0,
    vmax: float = 2.0,
    connect_daily=True
):
    """
    Reads a CSV with columns: [day, hour, price, action, reward].
    Creates a plot where:
      - x-axis = day + hour/24
      - y-axis = day-ahead price (euro/MWh)
      - color = action, in [-2..2] => negative (sell) = blue, positive (buy) = red
    Optionally filters the data to show only a specific range of days.

    Parameters:
      - csv_file: Path to the CSV file.
      - out_png: Path to save the plot.
      - day_start, day_end: Range of days to filter. If None, plots all days.
      - vmin, vmax: Range for color mapping of actions.
      - connect_daily: Whether to connect points within each day for continuity.
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Filter by day range if specified
    if day_start is not None and day_end is not None:
        df = df[(df["day"] >= day_start) & (df["day"] <= day_end)]
        if df.empty:
            print(f"No data available in the range of days {day_start} to {day_end}.")
            return

    # Create a fractional day axis: x_time = day + (hour - 1)/24
    df["x_time"] = df["day"] + (df["hour"] - 1) / 24.0

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Connect daily points if requested
    if connect_daily:
        for d in df["day"].unique():
            subset = df[df["day"] == d].sort_values("hour")
            plt.plot(subset["x_time"], subset["price"], color="gray", alpha=0.3, linewidth=1)

    # Scatter plot with color-coded actions
    sc = plt.scatter(
        df["x_time"],        # x-axis: fractional day
        df["price"],         # y-axis: price
        c=df["action"],      # color: action values
        cmap="RdBu_r",       # red-blue reversed colormap
        vmin=vmin, vmax=vmax,# color scale
        s=60,                # marker size
        edgecolors='k',      # black edges for clarity
        alpha=0.8
    )

    # Customize axes and title
    plt.xlabel("Days (fractional)", fontsize=12)
    plt.ylabel("Day-ahead price (euro/MWh)", fontsize=12)
    title = f"Illustration of Electricity Prices with Actions\n"
    if day_start is not None and day_end is not None:
        title += f"(Days {day_start} to {day_end})"
    plt.title(title, fontsize=14, pad=15)

    # Add colorbar for actions
    cb = plt.colorbar(sc)
    cb.set_label("Actions (MWh)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # Add grid for readability
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Plot saved to '{out_png}'.")
    plt.show()

plot_prices_and_actions("greedy_results_final.csv", out_png="my_actions_plot.png")
