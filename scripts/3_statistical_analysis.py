import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import os

df = pd.read_csv('data/online_retail_dataset.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice']  = df['Quantity'] * df['UnitPrice']

# Make sure output folder exists
os.makedirs('data', exist_ok=True)

print("Dataset loaded:", df.shape)

print("STATISTICAL ANALYSIS")

print("\n1. RFM ANALYSIS")

# Set analysis date (last date + 1 day)
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate RFM metrics for each customer
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo':   'nunique',                                  # Frequency
    'TotalPrice':  'sum'                                       # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Remove customers with negative monetary value
rfm = rfm[rfm['Monetary'] > 0]

print(f"RFM Metrics Calculated for {len(rfm)} customers")
print(f"\nRFM Summary Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

print(f"\nAverage Recency:   {rfm['Recency'].mean():.1f} days")
print(f"Average Frequency: {rfm['Frequency'].mean():.1f} orders")
print(f"Average Monetary:  £{rfm['Monetary'].mean():,.2f}")

print("\n2. CORRELATION ANALYSIS")

correlation_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
print("\nRFM Correlation Matrix:")
print(correlation_matrix)

print("\nKey Correlations:")
print(f"Frequency vs Monetary: {correlation_matrix.loc['Frequency', 'Monetary']:.3f}")
print(f"Recency vs Frequency:  {correlation_matrix.loc['Recency',   'Frequency']:.3f}")
print(f"Recency vs Monetary:   {correlation_matrix.loc['Recency',   'Monetary']:.3f}")

if correlation_matrix.loc['Frequency', 'Monetary'] > 0.5:
    print("→ Strong positive correlation: More frequent buyers spend more")
if correlation_matrix.loc['Recency', 'Monetary'] < -0.3:
    print("→ Negative correlation: Recent buyers tend to spend more")

print("\n3. K-MEANS CLUSTERING")

# Log transformation for Monetary (right-skewed)
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

# Standardize features
scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary_log']])

# Apply K-Means
n_clusters      = 4
kmeans          = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
rfm['Segment']  = kmeans.fit_predict(rfm_scaled)

print(f"\nCreated {n_clusters} customer segments using K-Means")
print("\nCluster Centers (scaled):")
print(kmeans.cluster_centers_)
print(f"\nInertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Segment characteristics
segment_summary = rfm.groupby('Segment').agg({
    'Recency':    'mean',
    'Frequency':  'mean',
    'Monetary':   'mean',
    'CustomerID': 'count'
}).round(2)

segment_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
print("\nSegment Characteristics:")
print(segment_summary)

# Name segments based on characteristics
segment_names = {}
for seg in range(n_clusters):
    rec  = segment_summary.loc[seg, 'Avg_Recency']
    freq = segment_summary.loc[seg, 'Avg_Frequency']
    mon  = segment_summary.loc[seg, 'Avg_Monetary']

    if freq > rfm['Frequency'].median() and mon > rfm['Monetary'].median():
        segment_names[seg] = 'Champions'
    elif rec < rfm['Recency'].median() and mon > rfm['Monetary'].median():
        segment_names[seg] = 'Loyal Customers'
    elif freq > rfm['Frequency'].median():
        segment_names[seg] = 'Potential Loyalists'
    else:
        segment_names[seg] = 'At Risk'

rfm['SegmentName'] = rfm['Segment'].map(segment_names)

print("\nSegment Distribution:")
print(rfm['SegmentName'].value_counts())

print("\n4. STATISTICAL HYPOTHESIS TESTING")

uk_orders      = df[df['Country'] == 'United Kingdom']['TotalPrice']
germany_orders = df[df['Country'] == 'Germany']['TotalPrice']

t_stat, p_value = stats.ttest_ind(uk_orders, germany_orders)
print(f"\nT-test: UK vs Germany Order Values")
print(f"UK Average:      £{uk_orders.mean():.2f}")
print(f"Germany Average: £{germany_orders.mean():.2f}")
print(f"T-statistic:     {t_stat:.3f}")
print(f"P-value:         {p_value:.4f}")

if p_value < 0.05:
    print("→ Significant difference in order values between UK and Germany")
else:
    print("→ No significant difference in order values")

print("\n5. DISTRIBUTION ANALYSIS")

stat, p_norm = stats.normaltest(rfm['Monetary'])
print(f"\nNormality Test for Monetary Values:")
print(f"Statistic: {stat:.3f}")
print(f"P-value:   {p_norm:.4f}")

if p_norm < 0.05:
    print("→ Monetary values are NOT normally distributed (expected for transaction data)")
    print("→ This justifies using log transformation for clustering")

print(f"\nSkewness: {rfm['Monetary'].skew():.3f}")
print(f"Kurtosis: {rfm['Monetary'].kurtosis():.3f}")

if rfm['Monetary'].skew() > 1:
    print("→ Highly right-skewed distribution (few high-value customers)")

# SAVE RESULTS

rfm.to_csv('data/rfm_segments.csv', index=False)
print("\n Results saved to: data/rfm_segments.csv")