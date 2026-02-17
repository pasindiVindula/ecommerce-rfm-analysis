import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# SUMMARY STATISTICS

# Load the dataset
df = pd.read_csv('data/online_retail_dataset.csv')

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Extract time features
df['Year']      = df['InvoiceDate'].dt.year
df['Month']     = df['InvoiceDate'].dt.month
df['Day']       = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour']      = df['InvoiceDate'].dt.hour

# Make sure output folder exists
os.makedirs('visualizations', exist_ok=True)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

print("EXPLORATORY DATA ANALYSIS - SUMMARY STATISTICS")

# 1. OVERALL DATASET STATISTICS
print("\n1. DATASET OVERVIEW:")
print(f"   Total Transactions: {len(df):,}")
print(f"   Unique Customers: {df['CustomerID'].nunique():,}")
print(f"   Unique Products: {df['StockCode'].nunique():,}")
print(f"   Countries Served: {df['Country'].nunique()}")
print(f"   Date Range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")

# 2. REVENUE STATISTICS
total_revenue = df['TotalPrice'].sum()
print(f"\n2. REVENUE METRICS:")
print(f"   Total Revenue: £{total_revenue:,.2f}")
print(f"   Average Transaction Value: £{df['TotalPrice'].mean():.2f}")
print(f"   Median Transaction Value: £{df['TotalPrice'].median():.2f}")
print(f"   Std Dev: £{df['TotalPrice'].std():.2f}")
print(f"   Min: £{df['TotalPrice'].min():.2f}")
print(f"   Max: £{df['TotalPrice'].max():.2f}")

# 3. QUANTITY STATISTICS
print(f"\n3. QUANTITY METRICS:")
print(f"   Average Quantity per Order: {df['Quantity'].mean():.2f}")
print(f"   Median Quantity: {df['Quantity'].median():.0f}")
print(f"   Total Items Sold: {df['Quantity'].sum():,}")

# 4. PRICE STATISTICS
print(f"\n4. UNIT PRICE STATISTICS:")
print(df['UnitPrice'].describe())

# 5. TOP COUNTRIES BY REVENUE
print(f"\n5. TOP 10 COUNTRIES BY REVENUE:")
country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
for i, (country, revenue) in enumerate(country_revenue.items(), 1):
    pct = (revenue / total_revenue) * 100
    print(f"   {i:2}. {country:20} £{revenue:>12,.2f} ({pct:5.2f}%)")

# 6. TOP PRODUCTS BY QUANTITY
print(f"\n6. TOP 10 PRODUCTS BY QUANTITY SOLD:")
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
for i, (product, qty) in enumerate(top_products.items(), 1):
    print(f"   {i:2}. {product[:50]:50} {qty:>6,} units")

# 7. TOP PRODUCTS BY REVENUE
print(f"\n7. TOP 10 PRODUCTS BY REVENUE:")
product_revenue = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
for i, (product, revenue) in enumerate(product_revenue.items(), 1):
    print(f"   {i:2}. {product[:50]:50} £{revenue:>10,.2f}")

# 8. TEMPORAL PATTERNS
print(f"\n8. MONTHLY SALES PATTERN:")
monthly_sales = df.groupby('Month')['TotalPrice'].sum().sort_index()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month_num, revenue in monthly_sales.items():
    print(f"   {months[month_num-1]:3}: £{revenue:>10,.2f}")

# 9. DAY OF WEEK ANALYSIS
print(f"\n9. SALES BY DAY OF WEEK:")
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sales = df.groupby('DayOfWeek')['TotalPrice'].sum()
for dow, revenue in dow_sales.items():
    print(f"   {day_names[dow]:9}: £{revenue:>10,.2f}")

# 10. CUSTOMER STATISTICS
print(f"\n10. CUSTOMER BEHAVIOR:")
customer_orders = df.groupby('CustomerID')['InvoiceNo'].nunique()
customer_spending = df.groupby('CustomerID')['TotalPrice'].sum()
print(f"   Average Orders per Customer: {customer_orders.mean():.2f}")
print(f"   Average Spending per Customer: £{customer_spending.mean():,.2f}")
print(f"   Most Active Customer Orders: {customer_orders.max()}")
print(f"   Highest Spending Customer: £{customer_spending.max():,.2f}")

# 11. CORRELATION MATRIX
print(f"\n11. CORRELATION ANALYSIS:")
numeric_df = df[['Quantity', 'UnitPrice', 'TotalPrice']].corr()
print(numeric_df)