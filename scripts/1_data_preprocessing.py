import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_csv('data/online_retail_dataset.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# 1. DATA TYPE CONVERSION
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print("\n✓ Date column converted to datetime")

# 2. HANDLE MISSING VALUES
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing CustomerID (if any)
df = df.dropna(subset=['CustomerID'])
print(f"\n✓ Missing values handled")

# 3. REMOVE DUPLICATES
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print(f"✓ Removed {duplicates} duplicate rows")

# 4. FEATURE ENGINEERING
# Calculate total price for each transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"✓ Created TotalPrice feature")

# 5. DATA VALIDATION
# Remove invalid transactions (cancelled orders, zero prices)
initial_count = len(df)
df = df[df['Quantity'] > 0]  # Remove returns for customer analysis
df = df[df['UnitPrice'] > 0]  # Remove zero-price items
final_count = len(df)
print(f"✓ Removed {initial_count - final_count} invalid transactions")

# 6. CATEGORICAL ENCODING
# Country is already categorical - no encoding needed for analysis
# But we can create a binary feature for UK vs International
df['is_UK'] = (df['Country'] == 'United Kingdom').astype(int)
print(f"✓ Created UK/International indicator")

# 7. TEMPORAL FEATURES
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
print(f"✓ Extracted temporal features")

# 8. FINAL DATASET INFO
print(f"\n✓ Preprocessing Complete!")
print(f"Final dataset shape: {df.shape}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Unique customers: {df['CustomerID'].nunique()}")
print(f"Total transactions: {len(df)}")

# Display first few rows
print("\nSample of preprocessed data:")
print(df.head())