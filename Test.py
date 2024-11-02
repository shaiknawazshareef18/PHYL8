# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Load the dataset
# Make sure the dataset is in the same directory or provide the full path to the dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('World Bank.CSV')

# 1. Data Exploration
# Display the first few rows of the dataset to get an overview
print("First row of the dataset:\n", df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics:\n", df.describe())

# Check for missing values in the dataset
print("\nMissing values:\n", df.isnull().sum())

# 2. Data Visualization: Clustered Column
# Visualize the Sum of Unemployed Rate (%) by Country using a Clustered Column
Unemployment_by_country = df.groupby('Country')['Unemployment Rate (%)'].sum()
Countries = Unemployment_by_country.index
Unemployment_sum = Unemployment_by_country.values

plt.figure(figsize=(12, 16))
plt.bar(Countries, Unemployment_sum)
plt.xlabel('country')
plt.ylabel('sum of unemployment Rate (%)')
plt.title('Sum of Unemployment Rate (%) by Country')
plt.show()

# Visualization: Scatter Plot of of GDP vs CO2 Emissions
plt.figure(figsize=(10, 6))
plt.scatter(df['GDP (USD)'], df['CO2 Emissions (metric tons per capita)'])
plt.xlabel('GDP (USD)')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.title('GDP vs CO2')
plt.show()

# Visualization: Distribution of Access to Electricity by country
plt.figure(figsize=(8, 6))
sns.boxplot(x='Access to Electricity (%)', y='Country', data=df)
plt.title('Access to Electricity by country')
plt.show()
