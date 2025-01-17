import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initialize the feature calculator
    # Load the dataset from the provided file path
    file_path = '../train.xlsx'
    data = pd.read_excel(file_path)

    # Display the first few rows of the dataset to understand its structure
    data.head()

    # Reshape the dataset into a long format
    data_long = data.melt(id_vars=["PRICES"],
                      var_name="Hour",
                      value_name="Price")

    # Convert the PRICES column to a datetime format for easier analysis
    data_long["PRICES"] = pd.to_datetime(data_long["PRICES"])

    # Extract hour as an integer for easier trend analysis
    data_long["Hour"] = data_long["Hour"].str.extract(r"(\d+)").astype(int)

    # Sort by date and hour
    data_long = data_long.sort_values(by=["PRICES", "Hour"]).reset_index(drop=True)

    # Display the first few rows of the reshaped data
    data_long.head()



    # Calculate price trends (momentum) and outliers

    # Shift the price column by 1 to calculate momentum (relative change from the previous hour)
    data_long["Previous_Price"] = data_long["Price"].shift(1)
    data_long["Momentum"] = (data_long["Price"] - data_long["Previous_Price"]) / data_long["Previous_Price"]

    # Compute rolling statistics (mean and standard deviation for Z-score calculation)
    window_size = 24  # One day rolling window
    data_long["Rolling_Mean"] = data_long["Price"].rolling(window=window_size).mean()
    data_long["Rolling_StdDev"] = data_long["Price"].rolling(window=window_size).std()

    # Calculate Z-score
    data_long["Z_Score"] = (data_long["Price"] - data_long["Rolling_Mean"]) / data_long["Rolling_StdDev"]

    # Identify potential outliers: Z-scores beyond +/-2
    data_long["Outlier"] = (data_long["Z_Score"].abs() > 2)

    # Drop rows with NaN values introduced by rolling calculations
    data_long_cleaned = data_long.dropna()

    # Display the first few rows with calculated features
    data_long_cleaned.head()



    # Plot prices with identified outliers
    plt.figure(figsize=(14, 8))
    plt.plot(data_long_cleaned["PRICES"], data_long_cleaned["Price"], label="Price", alpha=0.7)

    # Highlight outliers
    outliers = data_long_cleaned[data_long_cleaned["Outlier"]]
    plt.scatter(outliers["PRICES"], outliers["Price"], color="red", label="Outliers", zorder=5)

    # Add labels and legend
    plt.title("Price Trends with Outliers Highlighted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()



    # Plot Z-scores for additional context
    plt.figure(figsize=(14, 6))
    plt.plot(data_long_cleaned["PRICES"], data_long_cleaned["Z_Score"], label="Z-Score", color="purple", alpha=0.7)
    plt.axhline(2, color="red", linestyle="--", label="Outlier Threshold (+2)")
    plt.axhline(-2, color="red", linestyle="--", label="Outlier Threshold (-2)")

    # Add labels and legend
    plt.title("Z-Scores of Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    plt.show()



    from ProjectRL25.price_features.price_features import FeatureCalculator

    feature_calculator = FeatureCalculator(alpha=0.5, window_size=48, z_threshold=2.0)
    price_stream = data_long_cleaned["Price"].values
    hours = data_long_cleaned["Hour"].values

    # Compute features for each price
    computed_features = []
    for hour, price in zip(hours, price_stream):
        features = feature_calculator.calculate_features(price)
        features.update({"hour": hour, "Price": price, "Previous_Price": feature_calculator.price_history[-2] if len(
            feature_calculator.price_history) > 1 else None})
        computed_features.append(features)

    # Convert features to a DataFrame
    features_df = pd.DataFrame(computed_features)

    # Define binning logic using np.digitize
    # Momentum bins
    # Calculate dynamic edges based on quantiles
    momentum_quantiles = features_df["momentum"].quantile([0.33, 0.66]).values
    print(momentum_quantiles)
    momentum_edges = [-float("inf"), momentum_quantiles[0], momentum_quantiles[1], float("inf")]
    momentum_labels = ["negative", "neutral", "positive"]

    # Apply np.digitize for momentum binning
    momentum_bin = np.digitize(features_df["momentum"], bins=momentum_edges) - 1
    momentum_bin = np.clip(momentum_bin, 0, len(momentum_labels) - 1)

    # Assign the bins to the DataFrame
    features_df["momentum_bin"] = [momentum_labels[idx] for idx in momentum_bin]



    # Outlier bins
    outlier_edges = [-float('inf'), -1.5, 1.5, float('inf')]
    outlier_labels = ["negative_outlier", "non_outlier", "positive_outlier"]
    outlier_bin = np.digitize(features_df["z_score"], bins=outlier_edges) - 1
    outlier_bin = np.clip(outlier_bin, 0, len(outlier_labels) - 1)
    features_df["outlier_bin"] = [outlier_labels[idx] for idx in outlier_bin]

    # Display the resulting DataFrame
    print(features_df.head(100).to_string())

    # Iterate through price data to simulate actions
    simulation_data = []

    for idx, row in features_df.iterrows():
        current_price = row['Price']
        current_momentum = row['momentum']
        current_outlier = row['z_score']

        for action in [-1, 0, 1]:  # Simulate all possible actions: -1 (sell), 0 (hold), 1 (buy)
            reward = current_price * -action * 10  # Simplified reward calculation
            simulation_data.append({
                'Price': current_price,
                'Momentum': current_momentum,
                'Outlier': current_outlier,
                'Action': action,
                'Reward': reward
            })

    # Convert to DataFrame for analysis
    simulation_df = pd.DataFrame(simulation_data)
    print(simulation_df.head())

    correlation_matrix = simulation_df[['Momentum', 'Outlier', 'Reward']].corr()
    print("Correlation Matrix:\n", correlation_matrix)

    for action in [-1, 0, 1]:
        action_group = simulation_df[simulation_df['Action'] == action]
        print(f"Correlation Matrix for Action {action}:\n")
        print(action_group[['Momentum', 'Outlier', 'Reward']].corr())
        print("\n")

    grouped_rewards = simulation_df.groupby(['Action', 'Outlier'])['Reward'].mean()
    print(grouped_rewards)





