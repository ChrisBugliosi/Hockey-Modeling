import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

# Sample data for testing
data = {
    'xGoalsFor': [2.414, 1.617, 3.395, 2.424, 1.266, 2.887, 2.143, 2.105, 1.655, 2.215]
}
df = pd.DataFrame(data)

# Define rolling windows and EMA spans
rolling_windows = [5, 10, 15, 25, 40]
ema_spans = {
    'early': 3,
    'mid': 5,
    'late': 8
}

def weighted_moving_average(x, window):
    weights = np.arange(1, window + 1)
    return np.dot(x, weights) / weights.sum() if len(x) == window else np.nan

def calculate_moving_averages(df, columns, ema_spans, rolling_windows):
    result_df = pd.DataFrame()
    for column in columns:
        for window in rolling_windows:
            df[f'sma_{column}_rolling_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
            df[f'wma_{column}_rolling_{window}'] = df[column].rolling(window=window).apply(
                lambda x: weighted_moving_average(x, window), raw=False)
        df[f'ema_early_{column}'] = df[column].ewm(span=ema_spans['early'], adjust=False).mean()
        df[f'ema_mid_{column}'] = df[column].ewm(span=ema_spans['mid'], adjust=False).mean()
        df[f'ema_late_{column}'] = df[column].ewm(span=ema_spans['late'], adjust=False).mean()
    return df

# Test the function
columns_to_calculate = ['xGoalsFor']
result_df = calculate_moving_averages(df, columns_to_calculate, ema_spans, rolling_windows)

# Display the results for manual verification
print(result_df)

# Manually verify the calculations for a few rows
print("\nManual Calculation Verification")
sample_data = [2.414, 1.617, 3.395, 2.424, 1.266]
sma_manual = np.mean(sample_data)
wma_manual = weighted_moving_average(np.array(sample_data), 5)
ema_manual = 2.414  # Initial value for EMA
alpha = 2 / (3 + 1)
for value in sample_data[1:]:
    ema_manual = (value - ema_manual) * alpha + ema_manual

print(f"Manual SMA (5 periods): {sma_manual}")
print(f"Manual WMA (5 periods): {wma_manual}")
print(f"Manual EMA (span=3): {ema_manual}")









"""
import pandas as pd
import numpy as np

# Sample data for testing
data = {
    'xGoalsFor': [2.414, 1.617, 3.395, 2.424, 1.266, 2.887, 2.143, 2.105, 1.655, 2.215]
}
df = pd.DataFrame(data)

# Define rolling windows and EMA spans
rolling_window = 5
ema_span = 3

# Calculate SMA
df['sma_xGoalsFor_rolling_5'] = df['xGoalsFor'].rolling(window=rolling_window, min_periods=1).mean()

# Calculate WMA
def weighted_moving_average(x, window):
    weights = np.arange(1, window + 1)
    return np.dot(x, weights) / weights.sum() if len(x) == window else np.nan

df['wma_xGoalsFor_rolling_5'] = df['xGoalsFor'].rolling(window=rolling_window).apply(
    lambda x: weighted_moving_average(x, rolling_window), raw=False)

# Calculate EMA
df['ema_xGoalsFor'] = df['xGoalsFor'].ewm(span=ema_span, adjust=False).mean()

# Display the results for manual verification
print(df)

# Manually verify the calculations for a few rows
print("\nManual Calculation Verification")
sample_data = [2.414, 1.617, 3.395, 2.424, 1.266]
sma_manual = np.mean(sample_data)
wma_manual = weighted_moving_average(np.array(sample_data), rolling_window)
ema_manual = 2.414  # Initial value for EMA
alpha = 2 / (ema_span + 1)
for value in sample_data[1:]:
    ema_manual = (value - ema_manual) * alpha + ema_manual

print(f"Manual SMA (5 periods): {sma_manual}")
print(f"Manual WMA (5 periods): {wma_manual}")
print(f"Manual EMA (span=3): {ema_manual}")
"""