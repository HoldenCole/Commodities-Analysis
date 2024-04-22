import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Load the Excel file
file_path = r'C:\Users\19792\Documents\BrentCrudePrices.xlsx'

try:
    # Read the Excel file
    df = pd.read_excel(file_path, parse_dates=['Date'])
    # Sort by date if not already
    df.sort_values('Date', inplace=True)
    # Set the date as the index
    df.set_index('Date', inplace=True)
    # Ensure it's a daily frequency, forward fill any missing values
    df = df.asfreq('D').fillna(method='ffill')
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Check for stationarity
adf_test = adfuller(df['Price'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# If the p-value is > 0.05, we cannot reject the null hypothesis (the series is non-stationary)
# If the series is non-stationary, you may need to difference the series and re-run the test.

# Fit the ARIMA model
model = ARIMA(df['Price'], order=(1, 1, 1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Plot diagnostics
model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecast
forecast_steps = 365 * 3  # Forecasting 3 years into the future
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean}, index=forecast_index)

# Plot the historical data and forecast
plt.figure(figsize=(15, 7))
plt.plot(df['Price'], label='Historical Prices')
plt.plot(forecast_df['Forecast'], label='Forecast', linestyle='--')
plt.title('Brent Crude Prices Forecast with ARIMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()