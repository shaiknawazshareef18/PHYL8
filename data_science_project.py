# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Set the file path
file_path = r"C:\Users\siri chandana\Downloads\Python_mini\World Bank.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:\n", df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Convert columns to appropriate data types if necessary
df['Year'] = df['Year'].astype(int)

# Exploratory Data Analysis (EDA)
print("\nStatistical Summary:\n", df.describe())

# Correlation heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Life Expectancy
plt.figure(figsize=(8, 6))
sns.histplot(df['Life Expectancy'], kde=True, color='blue')
plt.title('Distribution of Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.show()

# Feature-target relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GDP (USD)', y='Life Expectancy', data=df)
plt.title('GDP vs. Life Expectancy')
plt.xlabel('GDP (USD)')
plt.ylabel('Life Expectancy')
plt.show()

# Define features (X) and target (y)
X = df[['GDP (USD)', 'Population', 'Unemployment Rate (%)', 
        'CO2Emissions (metric tons per capita)', 'Access to Electricity (%)']]
y = df['Life Expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (optional for Random Forest, but can help with interpretability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Visualize Actual vs. Predicted Life Expectancy
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs. Predicted Life Expectancy')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
