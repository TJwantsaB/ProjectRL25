import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_prices_and_actions(
    csv_file: str,
    out_png: str = "prices_actions_plot.png",
    day_start: int = 30,
    day_end: int = 40
):
    """
    Reads a CSV with columns [day,hour,price,action,reward].
    Plots (x=day+hour/24) vs. (y=price) with color-coded actions.
    
    - day_start, day_end: to filter only a certain range of days (inclusive).
    - We use a discrete colormap so that action=-1 => blue, action=0 => gray, action=1 => red.
    """
    # 1) Load data
    df = pd.read_csv(csv_file)
    
    # 2) Filter rows by day range
    df = df[(df["day"] >= day_start) & (df["day"] <= day_end)]
    if df.empty:
        print("No data in the specified day range!")
        return
    
    # 3) Create an x-axis that accounts for partial days:
    #    x = day + (hour-1)/24, so hour=1 => day + 0/24, hour=24 => day + 23/24
    df["x_time"] = df["day"] + (df["hour"] - 1) / 24.0

    # Prepare the scatter plot inputs
    x = df["x_time"]
    y = df["price"]
    c = df["action"]  # -1..0..+1

    # 4) Create a discrete colormap for actions: -1 => blue, 0 => gray, 1 => red
    #    We'll define bins for [-1.5, -0.5, 0.5, 1.5].
    #    Then -1 falls into first bin, 0 into second, 1 into third.
    my_cmap = ListedColormap(["blue", "gray", "red"])
    my_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], my_cmap.N)

    # 5) Plot
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(
        x, 
        y,
        c=c,
        cmap=my_cmap,
        norm=my_norm,
        s=40,
        alpha=0.8,
        edgecolors='k'  # small black edge around points (optional)
    )
    plt.xlabel("Days (fractional)")
    plt.ylabel("Day-ahead price (euro/MWh)")
    plt.title(f"Price vs. Time (Days {day_start} to {day_end})")

    # 6) Colorbar with discrete ticks
    cb = plt.colorbar(sc)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["Sell (-1)", "Nothing (0)", "Buy (1)"])
    cb.set_label("Actions")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Plot saved to '{out_png}'.")
    plt.show()


plot_prices_and_actions("greedy_results_final.csv", out_png="my_actions_plot.png")