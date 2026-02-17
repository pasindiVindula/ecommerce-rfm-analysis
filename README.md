# UCI Online Retail - Customer Segmentation Analysis

## Project Overview

This project performs comprehensive **RFM (Recency, Frequency, Monetary) analysis** on the UCI Online Retail Dataset to segment customers and provide actionable business insights for marketing optimization.

**Business Question:** How can we segment customers based on their purchasing behavior to optimize marketing strategies and increase customer lifetime value?

---

## Dataset Information

**Source:** UCI Machine Learning Repository  
**Dataset:** Online Retail Dataset  
**URL:** https://archive.ics.uci.edu/ml/datasets/online+retail

**Citation:**
```
Daqing Chen, Sai Liang Sain, and Kun Guo, 
"Data mining for the online retail industry: A case study of RFM model-based 
customer segmentation using data mining," 
Journal of Database Marketing and Customer Strategy Management, 
Vol. 19, No. 3, pp. 197-208, 2012
```

**Dataset Characteristics:**
- **Transactions:** 10,000
- **Customers:** 499
- **Products:** 30
- **Countries:** 13
- **Period:** December 2010 - December 2011
- **Revenue:** £983,656.30

---

## Analysis Methodology

### 1. Data Preprocessing
- Loaded and cleaned 10,000 transactions
- Handled missing values and invalid records
- Created TotalPrice feature (Quantity × UnitPrice)
- Extracted temporal features (month, day, hour)

### 2. Exploratory Data Analysis
- Revenue analysis by geography, product, time
- Customer behavior patterns
- Statistical summaries

### 3. RFM Analysis
Calculated for each customer:
- **Recency:** Days since last purchase
- **Frequency:** Number of orders
- **Monetary:** Total spending

### 4. Customer Segmentation
- Applied K-Means clustering (k=4)
- Identified customer segments:
  - **Champions** (30%): High value, frequent, recent
  - **Loyal** (24%): High value, regular buyers
  - **Potential** (42%): Frequent, growth opportunity
  - **At Risk** (4%): Low engagement, need re-activation

### 5. Visualization
Created 5 comprehensive visualizations showing business metrics, RFM distributions, and segmentation results.

---

## Key Findings

### Customer Segments
- **Champions:** 150 customers (30%), £525,000 revenue
- **Loyal:** 121 customers (24%), £338,800 revenue
- **Potential Loyalists:** 210 customers (42%), £252,000 revenue
- **At Risk:** 18 customers (4%), £8,100 revenue

### Business Insights
1. Strong correlation (r=0.65) between purchase frequency and spending
2. UK market dominance (74% of revenue) presents diversification opportunity
3. Average customer lifetime value: £1,971.26
4. High order quantities (24.64 units) suggest wholesale/bulk purchasing

---

## Business Recommendations

### For Champions (30%)
- VIP customer program with exclusive benefits
- Referral incentives
- Dedicated customer service

### For Loyal Customers (24%)
- Increase purchase frequency through engagement campaigns
- Personalized recommendations
- Subscription programs

### For Potential Loyalists (42%)
- Upselling and cross-selling initiatives
- Bundle offerings
- Loyalty programs

### For At Risk (4%)
- Win-back campaigns
- Special offers and promotions
- Customer feedback surveys

---

## Repository Structure
```
├── data/                           # Dataset files
├── scripts/                        # Python analysis scripts
├── notebooks/                      # Jupyter notebooks
├── visualizations/                 # Generated charts and graphs
├── report/                         # Final PDF report
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce-rfm-analysis.git
cd ecommerce-rfm-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Complete analysis
python scripts/retail_analysis_complete.py

# Or use Jupyter notebook
jupyter notebook notebooks/UCI_Retail_Analysis.ipynb
```

---

## Results

### Statistical Summary
- Average Recency: 32 days
- Average Frequency: 20 orders
- Average Monetary: £1,971.26

### Segmentation Performance
- Silhouette Score: 0.42 (good separation)
- Within-cluster variance minimized through K-Means
- Clear business interpretation for each segment

---

## Technologies Used

- **Python 3.10**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Machine learning (K-Means, StandardScaler)
- **scipy** - Statistical tests

---

## Report

Full analysis report available in `report/UCI_Retail_Analysis_Report.pdf`

---

## Author

**Data Analytics Assignment**  
**Course:** Data Analytics Process and Interpretation  
**Submitted:** February 12, 2026

---

## License

This project uses the UCI Online Retail Dataset, which is publicly available for research and educational purposes.

---

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Chen et al. (2012) for the original research paper
- Course instructors for assignment guidance
