# Microsoft Power BI Enterprise Dashboard Guide

This guide details how to construct the 4-page Bank Customer Churn Executive Report in **Power BI Desktop** using a premium dark fintech theme.

---

## 1. Data Connection & Setup

### A. Importing Datasets
Import the CSV files in the `/powerbi/` folder:
1. `churn_master.csv` (Main transactional customer table)
2. `geography_summary.csv`
3. `age_summary.csv`
4. `product_summary.csv`
5. `credit_score_summary.csv`
6. `model_metrics.csv`
7. `prediction_history.csv` (prediction logger file)

### B. SQLite Direct Connection (Alternative)
For live direct access:
1. Install the **SQLite ODBC Driver** on your system.
2. In Power BI Desktop, go to **Get Data** $\rightarrow$ **ODBC**.
3. Under Data Source Name (DSN), select your configured SQLite DSN (pointing to `/bank_churn.db`), or use connection string:
   `Driver={SQLite3 ODBC Driver};Database=c:\Users\admin\Desktop\myanalysisandml\bank_churn.db;`
4. Import the `customers` and `predictions` tables.

### C. Relationship Modeling (Model View)
Connect dimensions to the fact table:
- Link `geography_summary` [geography] $\rightarrow$ `churn_master` [geography] (1:many)
- Link `product_summary` [num_products] $\rightarrow$ `churn_master` [num_products] (1:many)
- Link `prediction_history` [customer_id] (if logged) or keep `prediction_history` as a standalone analytic dataset.

---

## 2. Power BI DAX Measures

Select the `churn_master` table and define these measures:

### Churn Rate %
```dax
Churn Rate % = 
DIVIDE(
    CALCULATE(COUNT(churn_master[customer_id]), churn_master[exited] = 1),
    COUNT(churn_master[customer_id]),
    0
)
```
*Format: Percentage (%), 2 decimal places.*

### Retention Rate %
```dax
Retention Rate % = 1 - [Churn Rate %]
```
*Format: Percentage (%), 2 decimal places.*

### Active Customer %
```dax
Active Customer % = 
DIVIDE(
    CALCULATE(COUNT(churn_master[customer_id]), churn_master[is_active_member] = 1),
    COUNT(churn_master[customer_id]),
    0
)
```
*Format: Percentage (%), 2 decimal places.*

### Avg Balance
```dax
Avg Balance = AVERAGE(churn_master[balance])
```
*Format: Currency (₹), 0 decimal places.*

### Avg Credit Score
```dax
Avg Credit Score = AVERAGE(churn_master[credit_score])
```
*Format: Decimal Number, 1 decimal place.*

### High Risk Customer Count (Prediction History)
```dax
High Risk Customer Count = 
CALCULATE(
    COUNT(prediction_history[prediction_id]), 
    prediction_history[risk_level] = "High"
)
```

---

## 3. Dashboard Page Layouts (Dark Fintech Theme)

### Page 1 — Executive Overview
*Provides high-level churn metrics for bank executives.*

- **Theme**: Dark navy background (`#080c14`), card background (`#111827`), accents in neon blue (`#3b82f6`) and coral red (`#ef4444`).
- **KPI Card Grid**:
  * Total Customers: `COUNT(churn_master[customer_id])`
  * Churned Customers: `CALCULATE(COUNT(churn_master[customer_id]), churn_master[exited]=1)`
  * Retained Customers: `CALCULATE(COUNT(churn_master[customer_id]), churn_master[exited]=0)`
  * Churn Rate %: `[Churn Rate %]` measure
  * Avg Balance: `[Avg Balance]` measure
  * Avg Credit Score: `[Avg Credit Score]` measure
- **Charts**:
  * **Churn by Geography**: Donut visual (`geography` as Legend, `[Churn Rate %]` as Value).
  * **Churn by Gender**: Clustered column visual (`gender` as X-axis, `[Churn Rate %]` as Y-axis).
  * **Churn by Age Group**: Column chart (`age_group` as X-axis, `[Churn Rate %]` as Y-axis).
  * **Churn Trend**: Line chart (`tenure` as X-axis, `[Churn Rate %]` as Y-axis).
- **Filters**: Slicers for `Geography`, `Gender`, and `Age Group`.

---

### Page 2 — Customer Analysis
*In-depth segmentation of customer behaviors.*

- **Visuals**:
  * **Customer Distribution by Geography**: Map visual (`geography` as Location, `Total Customers` as bubble size).
  * **Balance Distribution**: Histogram (binned `balance` as X-axis, customer count as Y-axis).
  * **Credit Score Distribution**: Clustered Column (`credit_score_range` as X-axis, customer count split by `exited` as Y-axis).
  * **Active vs Inactive Members**: Stacked bar visual (`is_active_member` as Y-axis, `[Churn Rate %]` as X-axis).
  * **Products vs Churn Rate**: Bar chart (`num_products` as X-axis, `[Churn Rate %]` as Y-axis).
  * **Salary vs Churn**: Scatter Plot (`estimated_salary` as X-axis, `balance` as Y-axis, colored by `exited`).
- **Drill-through**:
  * Configure right-click drill-through on any customer segment or geography to access detail tables.

---

### Page 3 — Machine Learning Insights
*Explains the predictive engine performance.*

- **Visuals**:
  * **Model Comparison Table**: Table visual displaying columns from `model_metrics.csv` (Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC).
  * **ROC-AUC Comparison**: Clustered bar chart (`Model` as Y-axis, `ROC-AUC` as X-axis).
  * **Precision vs Recall Comparison**: Scatter chart (`Precision` as X-axis, `Recall` as Y-axis, with model names as labels).
  * **Feature Importance Chart**: Clustered horizontal bar chart showing features ranked by mean importance.
- **Model Registry Context Cards**:
  * Selected Best Model: Card mapping the active model name (XGBoost).
  * Deployed model specs: Cards for Registry Version (`v1`), Dataset size (`10,000`), and active ROC-AUC (`0.8818`).

---

### Page 4 — Live Prediction Analytics
*Tracks real-time simulation entries and risk alerts.*

- **Visuals**:
  * **Prediction History**: Table visual (`timestamp`, `credit_score`, `geography`, `gender`, `age`, `churn_probability`, `risk_level`).
  * **High Risk Customers**: Table visual filtered to show only `risk_level = "High"`, highlighting customers with imminent churn warnings.
  * **Risk Distribution**: Pie visual (`risk_level` as Legend, count of predictions as Value).
  * **Churn Probability Distribution**: Column visual showing counts binned by churn probability ranges (e.g. 0-10%, 10-20%, etc.).
- **Filters**: Slicers for `Risk Level`, `Geography`, and `Date` (calendar).

---

## 4. Automatic Refresh Configuration

1. In the **Power BI Service**, configure the report to connect via an **On-premises Data Gateway** (Personal mode is sufficient for local databases).
2. Set up direct file paths or ODBC parameters in the gateway data source credentials.
3. In dataset settings, turn on **Scheduled Refresh** and configure it to run daily or weekly.
4. Running the python retraining script `06_retrain.py` automatically updates the local CSV sources, which updates the Power BI visuals upon the next refresh schedule or gateway sync event.
