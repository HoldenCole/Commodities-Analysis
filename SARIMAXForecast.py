import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings

# Define the inverse Box-Cox transformation function
def invboxcox(y, lam):
    if lam == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lam * y + 1) / lam)

# Load the Excel file
file_path = 'C:/Users/19792/Documents/BrentCrudePrices.xlsx'  # Corrected file path with forward slashes

# Read the data
df = pd.read_excel(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
df = df.asfreq('D')
df['Price'] = df['Price'].ffill()

# Check for missing values or very small values before transformation
if df['Price'].isnull().any():
    print("There are missing values in the 'Price' column.")

if (df['Price'] <= 0).any():
    print("There are non-positive values in the 'Price' column which cannot be transformed by Box-Cox.")

# Add a small constant to Price to avoid zero or negative values
df['Price'] = df['Price'] + 1e-10  # Ensure all values are positive

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Box-Cox transformation
    df['Price_transformed'], lam = boxcox(df['Price'])

    # Model fitting
    model = SARIMAX(df['Price_transformed'], order=(0, 1, 0), seasonal_order=(0, 1, 1, 12))
    results = model.fit()

    # Diagnostics plot
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    # Check if any warnings were raised
    for warning in w:
        print(f"Warning: {warning.message}")

# Forecasting
forecast_steps = 365 * 3  # Forecasting 3 years into the future
forecast_results = results.get_forecast(steps=forecast_steps)

# Get forecast and confidence intervals within one standard deviation
predicted_mean = forecast_results.predicted_mean
conf_int = forecast_results.conf_int(alpha=0.32)  # Approximately 1 std dev

# Inverse Box-Cox Transformation on forecast and confidence intervals
forecast_values = invboxcox(predicted_mean, lam)
lower_series = invboxcox(conf_int.iloc[:, 0], lam)
upper_series = invboxcox(conf_int.iloc[:, 1], lam)

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(df['Price'], label='Historical Prices')
plt.plot(predicted_mean.index, forecast_values, label='Forecast', color='orange')
plt.fill_between(predicted_mean.index, lower_series, upper_series, color='orange', alpha=0.3, label='Confidence Interval')
plt.title('Brent Crude Prices Forecast with Transformed SARIMAX')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()