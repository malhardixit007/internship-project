# generate_data.py
# Run this ONCE to generate your dataset. Don't run it again after that.

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Generate 2 years of daily sales data
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Simulate realistic sales with seasonality + noise
seasonality = 50 * np.sin(np.arange(n) * 2 * np.pi / 365)
trend = np.linspace(100, 180, n)  # slowly growing sales over time
noise = np.random.normal(0, 15, n)

data = {
    'date'      : dates,
    'sales'     : (trend + seasonality + noise).clip(min=30).astype(int),
    'price'     : np.random.uniform(20, 60, n).round(2),
    'discount'  : np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], n),
    'holiday'   : np.random.choice([0, 1], n, p=[0.93, 0.07]),
    'store_size': np.random.choice(['small', 'medium', 'large'], n),
    'category'  : np.random.choice(['Electronics', 'Clothing', 'Food', 'Sports'], n),
    'temperature': np.random.uniform(5, 42, n).round(1),  # weather effect
    'weekday'   : dates.day_name()
}

df = pd.DataFrame(data)

os.makedirs('data', exist_ok=True)
df.to_csv('data/sales_data.csv', index=False)
print(f"Dataset created with {len(df)} rows!")
print(df.head())