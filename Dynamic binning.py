import numpy as np
import pandas as pd

# Load the dataset from an Excel file
file_path = 'train.xlsx'  # Update with the correct file path
df = pd.read_excel(file_path, sheet_name='prices_2')  # Update sheet_name if needed

# Keep the original structure for later reconstruction
original_columns = df.columns
original_index = df.index

# Melt the dataset to consolidate hourly price columns into a single column
price_data = df.melt(id_vars=['PRICES'],
                     var_name='Hour',
                     value_name='Price')

# Convert 'PRICES' column to datetime for proper handling
price_data['PRICES'] = pd.to_datetime(price_data['PRICES'])

# Sort the data by timestamp for consistency
price_data = price_data.sort_values(by=['PRICES', 'Hour']).reset_index(drop=True)

# Add a column for the day of the week
price_data['Day_of_Week'] = price_data['PRICES'].dt.day_name()

# Calculate the average price for each day of the week
weekly_averages = price_data.groupby('Day_of_Week')['Price'].mean()

# Normalize prices using the weekly average for the corresponding day
price_data['Weekly_Normalized_Price'] = price_data.apply(
    lambda row: row['Price'] / weekly_averages[row['Day_of_Week']],
    axis=1
)

# Shorter rolling window for dynamic percentile-based bins
rolling_window_size = 48  # Example: last 48 hours
percentiles = [0, 25, 50, 75, 100]  # Quartiles
rolling_bins = []

# Recalculate bins dynamically
for i in range(len(price_data)):
    if i >= rolling_window_size:
        # Get the rolling window data
        rolling_data = price_data['Weekly_Normalized_Price'].iloc[i - rolling_window_size:i]
        # Compute percentiles for the rolling window
        bins = np.percentile(rolling_data, percentiles)
    else:
        # Default bins when insufficient data
        bins = [0, 0.25, 0.5, 0.75, 1.0]
    rolling_bins.append(bins)

# Map current prices to dynamically calculated bins
price_data['Dynamic_Bin_Index'] = [
    max(np.digitize(price_data['Weekly_Normalized_Price'].iloc[i], rolling_bins[i], right=True) - 1, 0)
    for i in range(len(price_data))
]

# Reshape back into the original structure
reshaped_bins = price_data.pivot(index='PRICES', columns='Hour', values='Dynamic_Bin_Index')

reshaped_bins = reshaped_bins.copy()

print(reshaped_bins.head())

# Ensure Hour columns are ordered from 1 to 24
reshaped_bins = reshaped_bins.rename(columns=lambda x: int(str(x).split()[-1]) if isinstance(x, str) else x)


# Reset the index and set it to range from 1 to the length of the DataFrame
reshaped_bins.index = range(1, len(reshaped_bins) + 1)

print(reshaped_bins.head())

# Save the reshaped bin indices back into a new Excel file
output_file_path = 'binned_train.xlsx'
reshaped_bins.to_excel(output_file_path)

print(f"Binned data saved to {output_file_path}.")
