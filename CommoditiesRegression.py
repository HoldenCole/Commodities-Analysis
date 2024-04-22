import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the Excel file
file_path = r'C:\Users\19792\Documents\BrentCrudePrices.xlsx'

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

if 'Price' not in df.columns:
    print("Error: The required 'Price' column is missing from the dataset.")
    exit()

if df.isnull().any().any():
    print("Warning: Missing values detected. Filling missing values with the mean of each column.")
    df.fillna(df.mean(), inplace=True)

# Create features
X = df[['Date_ordinal']]  # Using the date as the feature
y = df['Price']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Introducing polynomial features
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_scaled)

# Splitting the data into training and test sets
tss = TimeSeriesSplit(n_splits=5)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Alpha values for Lasso and Ridge
lasso_alphas = np.logspace(-6, 2, 100)
ridge_alphas = np.logspace(-6, 2, 100)

# Initialize and fit models with cross-validation
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': LassoCV(alphas=lasso_alphas, cv=tss, random_state=42),
    'Ridge Regression': RidgeCV(alphas=ridge_alphas, scoring='neg_mean_squared_error', cv=tss)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}")
    if hasattr(model, 'alpha_'):
        print(f"Best alpha for {name}: {model.alpha_}")

# Plot predictions and actual prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Actual Price', color='black', linewidth=2)
colors = {'Linear Regression': 'blue', 'Lasso Regression': 'red', 'Ridge Regression': 'green'}
for name, model in models.items():
    df[f'{name}_Prediction'] = model.predict(X_poly)
    plt.plot(df['Date'], df[f'{name}_Prediction'], label=f'{name}', linestyle='--', color=colors[name])
plt.title('Brent Crude Prices: Actual and Predicted by Different Models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Residual Plot for one of the models, e.g., Ridge Regression
residuals = y_test - models['Ridge Regression'].predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.title('Residual Plot for Ridge Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()