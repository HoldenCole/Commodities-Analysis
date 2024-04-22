import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Load the Excel file
file_path = r'C:\Users\19792\Documents\BrentCrudePrices.xlsx'

# Pre-processing
try:
    df = pd.read_excel(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Checking the dataset
if 'Price' not in df.columns:
    print("Error: The required 'Price' column is missing from the dataset.")
    exit()

# Filling missing values if any
if df.isnull().any().any():
    print("Warning: Missing values detected. Filling missing values with the mean of each column.")
    df.fillna(df.mean(), inplace=True)

# Splitting the dataset
X = df[['Date_ordinal']]  # Features
y = df['Price']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions
y_pred_test = linear_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Linear Regression Test MSE: {mse_test:.2f}")
print(f"Linear Regression Test R^2: {r2_test:.2f}")

# Extending the date range by 3 years for forecasting
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=3 * 365, freq='D').tolist()

# Predicting future prices
future_df = pd.DataFrame(future_dates, columns=['Date'])
future_df['Date_ordinal'] = future_df['Date'].map(pd.Timestamp.toordinal)
future_predictions = linear_model.predict(future_df[['Date_ordinal']])

# Combining historical and future predictions
combined_dates = pd.concat([df['Date'], future_df['Date']])
combined_predictions = np.concatenate([linear_model.predict(X), future_predictions])

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(df['Date'], df['Price'], label='Historical Prices', color='black')
plt.plot(combined_dates, combined_predictions, label='Linear Regression Predictions', linestyle='--', color='blue')
plt.title('Brent Crude Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()