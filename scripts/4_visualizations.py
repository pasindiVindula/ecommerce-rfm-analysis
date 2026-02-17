import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import os

# STEP 1: LOAD DATA

df = pd.read_csv('data/online_retail_dataset.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice']  = df['Quantity'] * df['UnitPrice']
df['Month']       = df['InvoiceDate'].dt.month

os.makedirs('visualizations', exist_ok=True)

print("Dataset loaded:", df.shape)

# STEP 2: BUILD RFM

analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,
    'InvoiceNo':   'nunique',
    'TotalPrice':  'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# K-Means segmentation
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary_log']])

kmeans         = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Name segments
segment_summary = rfm.groupby('Segment').agg({
    'Recency':   'mean',
    'Frequency': 'mean',
    'Monetary':  'mean'
}).round(2)

segment_names = {}
for seg in range(4):
    rec  = segment_summary.loc[seg, 'Recency']
    freq = segment_summary.loc[seg, 'Frequency']
    mon  = segment_summary.loc[seg, 'Monetary']

    if freq > rfm['Frequency'].median() and mon > rfm['Monetary'].median():
        segment_names[seg] = 'Champions'
    elif rec < rfm['Recency'].median() and mon > rfm['Monetary'].median():
        segment_names[seg] = 'Loyal Customers'
    elif freq > rfm['Frequency'].median():
        segment_names[seg] = 'Potential Loyalists'
    else:
        segment_names[seg] = 'At Risk'

rfm['SegmentName'] = rfm['Segment'].map(segment_names)

print("RFM segments built:", rfm['SegmentName'].value_counts().to_dict())

# STEP 3: VISUALIZATIONS

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\nCreating visualizations...")

# FIGURE 1: Business Overview (2x2)

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('UCI Online Retail - Business Overview', fontsize=16, fontweight='bold')

# 1.1 Daily Revenue Trend
daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum()
axes[0, 0].plot(daily_revenue.index, daily_revenue.values,
                linewidth=1, color='steelblue', alpha=0.8)
axes[0, 0].set_xlabel('Date',         fontweight='bold')
axes[0, 0].set_ylabel('Revenue (£)',  fontweight='bold')
axes[0, 0].set_title('Daily Revenue Trend')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(alpha=0.3)

# 1.2 Top 10 Countries
top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
axes[0, 1].barh(range(len(top_countries)), top_countries.values, color='lightcoral')
axes[0, 1].set_yticks(range(len(top_countries)))
axes[0, 1].set_yticklabels(top_countries.index)
axes[0, 1].set_xlabel('Revenue (£)', fontweight='bold')
axes[0, 1].set_title('Top 10 Countries by Revenue')
axes[0, 1].grid(axis='x', alpha=0.3)

# 1.3 Order Value Distribution
valid_prices = df[df['TotalPrice'] > 0]['TotalPrice']
axes[1, 0].hist(valid_prices, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Order Value (£)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency',       fontweight='bold')
axes[1, 0].set_title('Order Value Distribution')
axes[1, 0].set_xlim(0, 500)
axes[1, 0].axvline(valid_prices.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: £{valid_prices.median():.2f}')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 1.4 Monthly Sales
monthly     = df.groupby('Month')['TotalPrice'].sum()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[1, 1].bar(monthly.index, monthly.values,
               color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1, 1].set_xlabel('Month',        fontweight='bold')
axes[1, 1].set_ylabel('Revenue (£)',  fontweight='bold')
axes[1, 1].set_title('Monthly Sales Pattern')
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].set_xticklabels([month_names[i-1] for i in monthly.index], rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/figure1_business_overview.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved: visualizations/figure1_business_overview.png")
plt.close()

# FIGURE 2: RFM Distributions (1x3)

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('RFM Analysis - Distribution of Customer Metrics',
              fontsize=16, fontweight='bold')

# 2.1 Recency
axes[0].hist(rfm['Recency'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Recency (days)',       fontweight='bold')
axes[0].set_ylabel('Number of Customers', fontweight='bold')
axes[0].set_title('Recency Distribution')
axes[0].axvline(rfm['Recency'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {rfm["Recency"].mean():.0f} days')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2.2 Frequency
axes[1].hist(rfm['Frequency'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Frequency (orders)',   fontweight='bold')
axes[1].set_ylabel('Number of Customers', fontweight='bold')
axes[1].set_title('Frequency Distribution')
axes[1].axvline(rfm['Frequency'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {rfm["Frequency"].mean():.1f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 2.3 Monetary
axes[2].hist(rfm['Monetary'], bins=30, color='gold', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Monetary Value (£)',   fontweight='bold')
axes[2].set_ylabel('Number of Customers', fontweight='bold')
axes[2].set_title('Monetary Distribution')
axes[2].set_xlim(0, 5000)
axes[2].axvline(rfm['Monetary'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: £{rfm["Monetary"].mean():.0f}')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/figure2_rfm_distributions.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved: visualizations/figure2_rfm_distributions.png")
plt.close()

# FIGURE 3: Customer Segmentation (1x2)

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Customer Segmentation Results', fontsize=16, fontweight='bold')

# 3.1 Pie chart
segment_counts = rfm['SegmentName'].value_counts()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
axes[0].pie(segment_counts.values,
            labels=segment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontweight': 'bold', 'fontsize': 11})
axes[0].set_title('Customer Segment Distribution')

# 3.2 Revenue by segment
segment_revenue = rfm.groupby('SegmentName')['Monetary'].sum().sort_values()
axes[1].barh(range(len(segment_revenue)), segment_revenue.values,
             color=colors, edgecolor='black')
axes[1].set_yticks(range(len(segment_revenue)))
axes[1].set_yticklabels(segment_revenue.index, fontweight='bold')
axes[1].set_xlabel('Total Revenue (£)', fontweight='bold')
axes[1].set_title('Revenue Contribution by Segment')
axes[1].grid(axis='x', alpha=0.3)

for i, v in enumerate(segment_revenue.values):
    axes[1].text(v + 5000, i, f'£{v:,.0f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/figure3_customer_segments.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved: visualizations/figure3_customer_segments.png")
plt.close()

# FIGURE 4: 3D RFM Scatter

fig4 = plt.figure(figsize=(12, 8))
ax   = fig4.add_subplot(111, projection='3d')

sample  = rfm.sample(min(500, len(rfm)), random_state=42)
scatter = ax.scatter(sample['Recency'],
                     sample['Frequency'],
                     sample['Monetary'],
                     c=sample['Segment'],
                     cmap='viridis',
                     s=50, alpha=0.6,
                     edgecolors='black', linewidth=0.5)

ax.set_xlabel('Recency (days)',    fontweight='bold', labelpad=10)
ax.set_ylabel('Frequency (orders)',fontweight='bold', labelpad=10)
ax.set_zlabel('Monetary (£)',      fontweight='bold', labelpad=10)
ax.set_title('3D RFM Customer Segmentation', fontsize=14, fontweight='bold', pad=20)

plt.colorbar(scatter, label='Segment', pad=0.1, shrink=0.8)
plt.tight_layout()
plt.savefig('visualizations/figure4_rfm_3d.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved: visualizations/figure4_rfm_3d.png")
plt.close()

# FIGURE 5: Correlation Heatmap

fig5, ax         = plt.subplots(figsize=(8, 6))
correlation_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()

sns.heatmap(correlation_matrix,
            annot=True, fmt='.3f',
            cmap='coolwarm', center=0,
            square=True, linewidths=2,
            cbar_kws={"shrink": 0.8},
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})

plt.title('RFM Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/figure5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Figure 5 saved: visualizations/figure5_correlation_heatmap.png")
plt.close()