import numpy as np

class FeatureCalculator:
    def __init__(self, alpha=0.1, window_size=10, z_threshold=2.0):
        self.alpha = alpha  # Smoothing factor for EMA
        self.window_size = window_size  # Rolling window size for standard deviation and momentum
        self.z_threshold = z_threshold  # Threshold for clipping based on Z-score
        self.price_history = []  # List to store recent prices
        self.ema = None  # Exponential Moving Average
        self.last_valid_mean = None  # Last valid rolling mean
        self.last_valid_std_dev = None  # Last valid rolling standard deviation

    def calculate_features(self, current_price):
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)  # Maintain rolling window

        # Calculate rolling statistics excluding outliers
        if len(self.price_history) >= 2:
            rolling_mean = np.mean(self.price_history)
            rolling_std_dev = np.std(self.price_history)

            # Detect outliers in the rolling window
            z_scores = [(price - rolling_mean) / rolling_std_dev if rolling_std_dev > 0 else 0 for price in self.price_history]
            valid_prices = [price for price, z in zip(self.price_history, z_scores) if abs(z) <= self.z_threshold]

            # Recalculate rolling stats using only valid prices
            if valid_prices:
                rolling_mean = np.mean(valid_prices)
                rolling_std_dev = np.std(valid_prices)

        else:
            rolling_mean = current_price
            rolling_std_dev = 0  # Not enough data for standard deviation

        # Detect and clip outliers using the last valid statistics
        clipped_price = current_price  # Default to the current price
        z_score = 0  # Default Z-score
        if rolling_std_dev > 0:
            z_score = (current_price - rolling_mean) / rolling_std_dev
            if abs(z_score) > self.z_threshold:
                # Use the last valid mean and std dev for clipping
                if self.last_valid_mean is not None and self.last_valid_std_dev is not None:
                    upper_bound = self.last_valid_mean + self.z_threshold * self.last_valid_std_dev
                    lower_bound = self.last_valid_mean - self.z_threshold * self.last_valid_std_dev
                else:
                    upper_bound = rolling_mean + self.z_threshold * rolling_std_dev
                    lower_bound = rolling_mean - self.z_threshold * rolling_std_dev
                clipped_price = max(min(current_price, upper_bound), lower_bound)
            else:
                # Update last valid stats if the current point is not an outlier
                self.last_valid_mean = rolling_mean
                self.last_valid_std_dev = rolling_std_dev
        else:
            # Not enough data to calculate Z-score; update valid stats
            self.last_valid_mean = rolling_mean
            self.last_valid_std_dev = rolling_std_dev

        # Log the Z-score and clipping for debugging
        # print(f"Price: {current_price}, Z-Score: {z_score:.2f}, Clipped Price: {clipped_price}")

        # Calculate EMA using the clipped price
        if self.ema is None:
            self.ema = clipped_price  # Initialize EMA with the first price
        else:
            self.ema = self.alpha * clipped_price + (1 - self.alpha) * self.ema

        # Calculate momentum (relative change over the rolling window)
        if len(self.price_history) > 1:
            if len(self.price_history) > 1:
                # Use only the most recent 5 prices (or fewer if history is short)
                lookback = min(3, len(self.price_history))
                history = self.price_history[-lookback:]

                # Exponential weights for more responsiveness
                weights = np.exp(np.linspace(0, 1, len(history)))

                # Weighted average
                weighted_avg = np.average(history, weights=weights)

                # Calculate momentum
                momentum = (current_price - weighted_avg) / weighted_avg
        else:
            momentum = 0  # No momentum if there's only one price

        # Return calculated features
        return {
            "ema": self.ema,
            "z_score": z_score,
            "momentum": momentum,
            "recent_std_dev": self.last_valid_std_dev,
            "clipped_price": clipped_price,
        }

# Example Usage
if __name__ == "__main__":
    # Initialize the feature calculator
    feature_calculator = FeatureCalculator(alpha=0.2, window_size=5, z_threshold=1.5)

    # Example price stream
    price_stream = [100, 102, 101, 105, 2500, 108]

    # Compute features for each price
    for price in price_stream:
        features = feature_calculator.calculate_features(price)
        print(f"Price: {price}, Features: {features}")