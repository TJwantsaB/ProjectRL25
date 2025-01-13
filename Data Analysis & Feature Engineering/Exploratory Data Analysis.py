# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = '../train.xlsx'
df = pd.read_excel(file_path, sheet_name='prices_2')

# Step 1: Basic Information and Structure
print("Dataset Overview:")
print(df.info())
print("\nMissing Values Summary:")
print(df.isnull().sum())

# Step 2: Statistical Summary
print("\nStatistical Summary:")
print(df.describe().to_string())

# Step 3: Time-Series Preparation
# Convert the PRICES column to datetime and set it as index
df['PRICES'] = pd.to_datetime(df['PRICES'])
df.set_index('PRICES', inplace=True)
df.columns = [col.replace('Hour ', '') if 'Hour' in col else col for col in df.columns]


import matplotlib.pyplot as plt

# Add a column for the day of the week
df['Day_of_Week'] = df['PRICES'].dt.day_name()

# Calculate the average price for each day of the week
average_prices_per_day = df.groupby('Day_of_Week')['Price'].mean()

# Reorder the days of the week for a proper display
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
average_prices_per_day = average_prices_per_day.reindex(ordered_days)

# Plot the average prices per day
plt.figure(figsize=(10, 6))
plt.plot(average_prices_per_day.index, average_prices_per_day.values, marker='o', linestyle='-', color='blue', label='Day-of-Week Average')
plt.title('Average Prices by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print the average prices for each day
print("Average Prices by Day of the Week:")
print(average_prices_per_day)

hourly_flat = df.iloc[:, 1:].values.flatten()

# Create a plot with a logarithmic scale
plt.figure(figsize=(15, 8))
plt.plot(hourly_flat, label='Hourly Prices', alpha=0.7, color='blue')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Hourly Prices Across the Year (Logarithmic Scale)')
plt.xlabel('Hour Instance')
plt.ylabel('Price (Log Scale)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Hourly Prices')
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df= df.where((df >= Q1 - 1.5 * IQR) & (df <= Q3 + 1.5 * IQR), np.nan)

hourly_flat = df.iloc[:, 1:].values.flatten()

# Create a plot with a logarithmic scale
plt.figure(figsize=(15, 8))
plt.plot(hourly_flat, label='Hourly Prices', alpha=0.7, color='blue')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Hourly Prices Across the Year (Logarithmic Scale)')
plt.xlabel('Hour Instance')
plt.ylabel('Price (Log Scale)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Hourly Prices (Outliers Removed)')
plt.show()

# Step 3: Calculate Averages
# Hourly Average (mean across all days for each hour)
hourly_avg = df.mean(axis=0)[1:]  # Exclude the first column (date)

# Daily Average (mean across all hourly values for each day)
daily_avg = df.mean(axis=1)

# Weekly Average (resampling to weekly frequency)
weekly_avg = daily_avg.resample('W').mean()

# Monthly Average (resampling to monthly frequency)
monthly_avg = daily_avg.resample('M').mean()



# Plot 1: Hourly Average
plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, label='Hourly Average', color='orange', linewidth=2)
plt.title('Hourly Average Energy Prices')
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.xticks(rotation=45)  # Rotate x-ticks for better readability
plt.legend()
plt.show()

# Plot 2: Daily Average
plt.figure(figsize=(12, 6))
plt.plot(daily_avg, label='Daily Average', color='blue', linewidth=1)
plt.title('Daily Average Energy Prices')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()


# Plot 3: Weekly Average
plt.figure(figsize=(12, 6))
plt.plot(weekly_avg, label='Weekly Average', color='green', linewidth=2)
plt.title('Weekly Average Energy Prices')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()


# Plot 4: Monthly Average
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg, label='Monthly Average', color='red', linewidth=2)
plt.title('Monthly Average Energy Prices')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()

# Step 1: Compute Hourly Averages
hourly_avg = df.iloc[:, 1:].mean(axis=0)  # Exclude the date column
hour_labels = [f'Hour {i+1}' for i in range(len(hourly_avg))]  # Adjust labels dynamically
hourly_avg.index = hour_labels  # Rename index with dynamic labels


# Step 2: Compute Day-of-Week Averages
df['DayOfWeek'] = df.index.dayofweek  # Monday = 0, ..., Sunday = 6
day_of_week_avg = df.groupby('DayOfWeek').mean().mean(axis=1)
day_of_week_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Step 3: Compute Monthly Averages
df['Month'] = df.index.month  # Extract month
monthly_avg = df.groupby('Month').mean().mean(axis=1)
month_labels = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

# Step 4: Plot Hourly Averages
plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, label='Hourly Average', marker='o', color='orange')
plt.title('Average Prices by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Step 5: Plot Day-of-Week Averages
plt.figure(figsize=(12, 6))
plt.plot(day_of_week_labels, day_of_week_avg.values, label='Day-of-Week Average', marker='o', color='blue')
plt.title('Average Prices by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Plot Monthly Averages
plt.figure(figsize=(12, 6))
plt.plot(month_labels, monthly_avg.values, label='Monthly Average', marker='o', color='green')
plt.title('Average Prices by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# Step 2: Calculate Descriptive Statistics for Each Average
stats = {
    'Hourly': {
        'Mean': hourly_avg.mean(),
        'Variance': hourly_avg.var(),
        'Standard Deviation': hourly_avg.std(),
        'Minimum': hourly_avg.min(),
        'Maximum': hourly_avg.max(),
        'Median': hourly_avg.median(),
        'Range': hourly_avg.max() - hourly_avg.min(),
    },
    'Daily': {
        'Mean': daily_avg.mean(),
        'Variance': daily_avg.var(),
        'Standard Deviation': daily_avg.std(),
        'Minimum': daily_avg.min(),
        'Maximum': daily_avg.max(),
        'Median': daily_avg.median(),
        'Range': daily_avg.max() - daily_avg.min(),
    },
    'Weekly': {
        'Mean': weekly_avg.mean(),
        'Variance': weekly_avg.var(),
        'Standard Deviation': weekly_avg.std(),
        'Minimum': weekly_avg.min(),
        'Maximum': weekly_avg.max(),
        'Median': weekly_avg.median(),
        'Range': weekly_avg.max() - weekly_avg.min(),
    },
    'Monthly': {
        'Mean': monthly_avg.mean(),
        'Variance': monthly_avg.var(),
        'Standard Deviation': monthly_avg.std(),
        'Minimum': monthly_avg.min(),
        'Maximum': monthly_avg.max(),
        'Median': monthly_avg.median(),
        'Range': monthly_avg.max() - monthly_avg.min(),
    },
}

# Step 3: Convert to DataFrame
stats_df = pd.DataFrame(stats).T

print(stats_df.to_string())

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# Step 1: Calculate Volatility
daily_volatility = df.std(axis=1)  # Calculate daily volatility as the standard deviation of hourly prices
weekly_volatility = daily_volatility.resample('W').mean()  # Weekly volatility
monthly_volatility = daily_volatility.resample('M').mean()  # Monthly volatility

# Step 2: Plot Volatility Over Time
plt.figure(figsize=(12, 6))
plt.plot(daily_volatility, label='Daily Volatility', color='blue', alpha=0.6)
plt.plot(weekly_volatility, label='Weekly Volatility', color='orange', linewidth=2)
plt.plot(monthly_volatility, label='Monthly Volatility', color='green', linewidth=2)
plt.title('Volatility Over Time')
plt.xlabel('Date')
plt.ylabel('Volatility (Standard Deviation)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Step 3: Centralization of Volatility (Histogram and KDE)
plt.figure(figsize=(10, 6))
sns.histplot(daily_volatility, kde=True, bins=30, color='purple', alpha=0.7)
plt.title('Distribution of Daily Volatility')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Step 4: Autocorrelation of Volatility
plt.figure(figsize=(10, 6))
plot_acf(daily_volatility.dropna(), lags=30, alpha=0.05)
plt.title('Autocorrelation of Daily Volatility')
plt.xlabel('Lags (Days)')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.show()

# Step 5: Volatility Predicting Volatility
# Scatter plot of lagged volatility
volatility_lagged = daily_volatility.shift(1)  # Lagged volatility
plt.figure(figsize=(10, 6))
plt.scatter(volatility_lagged, daily_volatility, alpha=0.5, color='darkblue')
plt.title('Lagged Volatility vs Current Volatility')
plt.xlabel('Lagged Volatility')
plt.ylabel('Current Volatility')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate correlation between lagged and current volatility
correlation = daily_volatility.corr(volatility_lagged)
print(f"Correlation between lagged and current volatility: {correlation:.2f}")


