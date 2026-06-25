# Bank Customer Churn Analysis and Prediction System

A production-grade, end-to-end analytics and machine learning pipeline for retail banking. This system covers synthetic data engineering (SQL), predictive modeling (Python/XGBoost), business intelligence (Power BI ready files), and an interactive executive web dashboard.

---

## 1. Project Architecture

```text
  +-----------------------------------------------------------+
  |                        DATA LAYER                         |
  |  +-------------------+      +--------------------------+  |
  |  |  generate_data.py | ---> | SQLite (bank_churn.db)   |  |
  |  +-------------------+      +--------------------------+  |
  |                                 |                         |
  |                                 v                         |
  |                     +--------------------------+          |
  |                     |  churn_analysis.sql      |          |
  |                     +--------------------------+          |
  +---------------------------------|-------------------------+
                                    |
            +-----------------------+-----------------------+
            v                                               v
  +-----------------------------------+           +-----------------------------------+
  |          ML ENGINE (Python)       |           |        BUSINESS INTELLIGENCE      |
  |  +-----------------------------+  |           |  +-----------------------------+  |
  |  | 01_eda.py                   |  |           |  | export_powerbi.py           |  |
  |  |   -> Visualizations         |  |           |  |   -> /powerbi/ *.csv        |  |
  |  +-----------------------------+  |           |  +-----------------------------+  |
  |  +-----------------------------+  |           |  +-----------------------------+  |
  |  | 02_preprocessing.py         |  |           |  | powerbi_guide.md            |  |
  |  |   -> preprocessor.pkl       |  |           |  |   -> Power BI Visuals Setup |  |
  |  +-----------------------------+  |           |  +-----------------------------+  |
  |  +-----------------------------+  |           +-----------------------------------+
  |  | 03_train_models.py          |  |
  |  |   -> model_comparison.csv   |  |
  |  +-----------------------------+  |
  |  +-----------------------------+  |
  |  | 04_best_model.py            |  |
  |  |   -> best_model.pkl         |  |
  |  +-----------------------------+  |
  |  +-----------------------------+  |
  |  | 05_predict.py               |  |
  |  |   -> Risk Scoring Function  |  |
  |  +-----------------------------+  |
  +-----------------|-----------------+
                    |
                    v
  +-----------------------------------------------------------+
  |                   EXECUTIVE DASHBOARD                     |
  |  +-----------------------------------------------------+  |
  |  | Flask App Backend (app.py)                          |  |
  |  +-----------------------------------------------------+  |
  |  +-----------------------------------------------------+  |
  |  | Chart.js + HTML Frontend (templates/index.html)     |  |
  |  +-----------------------------------------------------+  |
  +-----------------------------------------------------------+
```

---

## 2. Directory Structure

```text
myanalysisandml/
├── bank_churn.db              # SQLite Database File
├── requirements.txt           # Python Project Dependencies
├── README.md                  # Project Main Documentation
├── sql/
│   ├── schema.sql             # SQL Schema definition
│   ├── seed_data.sql          # SQL statements populating 10,000 customers
│   └── churn_analysis.sql     # Analytical SQL queries
├── scripts/
│   ├── generate_data.py       # Python script generating the synthetic SQL seed
│   ├── export_reports.py      # SQLite query to data/exports CSV extractor
│   ├── export_powerbi.py      # Exporter for Power BI dashboard files
│   └── test_predict.py        # API Verification Scratch script
├── data/
│   └── exports/               # Analytical SQL Query Output CSVs
├── ml/
│   ├── temp_data/             # Pre-split CSV splits (X_train, y_train, etc.)
│   ├── charts/                # EDA & Model Evaluation generated plots
│   ├── preprocessor.pkl       # Fitted sklearn preprocessor object
│   ├── best_model.pkl         # Fine-tuned XGBoost model pkl
│   ├── model_comparison.csv   # Validation metrics comparing 4 classifiers
│   ├── 01_eda.py              # Exploratory Data Analysis Runner
│   ├── 02_preprocessing.py    # Standard Scaling & One Hot Encoder Pipeline
│   ├── 03_train_models.py     # Training suite comparing 4 models
│   ├── 04_best_model.py       # GridSearchCV Fine-tuning & Permutation Importance
│   └── 05_predict.py          # Inference script containing sample personas
├── dashboard/
│   ├── app.py                 # Flask Server serving stats & prediction routes
│   └── templates/
│       └── index.html         # Premium dark navy Chart.js dashboard UI
└── powerbi/
    ├── churn_master.csv       # Power BI transactional dataset
    ├── geography_summary.csv  # Geography analytics export
    ├── age_summary.csv        # Binned Age groups export
    ├── product_summary.csv    # Products vs Churn rate export
    ├── model_metrics.csv      # ML benchmarking export
    └── powerbi_guide.md       # Step-by-step Report building guidelines
```

---

## 3. Setup and Installation

### Prerequisites
- Python 3.9 or higher installed.

### Setup Instructions
1. Clone this repository or open the workspace directory.
2. Initialize virtual environment (optional, but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Execution Guide

### Phase 1 — SQL & Database Generation
Generate the synthetic database and export analytical reports:
```bash
python scripts/generate_data.py
python scripts/export_reports.py
```
This initializes `bank_churn.db`, populates it with 10,000 customers (churn rate ~12.55%), and saves report CSVs into `data/exports/`.

### Phase 2 — Machine Learning Pipeline
Run the ML modeling pipeline in sequence:
```bash
python ml/01_eda.py
python ml/02_preprocessing.py
python ml/03_train_models.py
python ml/04_best_model.py
python ml/05_predict.py
```
- `01_eda.py` saves analytical plots under `ml/charts/`.
- `03_train_models.py` trains and evaluates Logistic Regression, Random Forest, SVM, and XGBoost.
- `04_best_model.py` fine-tunes the best model via Grid Search, outputs a confusion matrix, and plots permutation feature importances.
- `05_predict.py` executes predictions on test mock profiles.

### Phase 3 — Flask Dashboard
Launch the local web dashboard:
```bash
python dashboard/app.py
```
Once active, navigate to **`http://localhost:5000`** in your browser.

### Phase 4 — Power BI Exports
Compile and export datasets for Power BI Desktop:
```bash
python scripts/export_powerbi.py
```
Review instructions in `powerbi/powerbi_guide.md` to design reports using Power BI Desktop.

---

## 5. Machine Learning Benchmarks

The database contains **10,000 customers** with an overall churn rate of **20.37%** (7,963 retained, 2,037 churned). 

The ML models trained on a stratified 80/20 train/test split yielded the following validation metrics:

| Model Class | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 81.35% | 60.12% | 24.82% | 35.13% | 0.7648 |
| Random Forest | 86.85% | 77.48% | 49.88% | 60.69% | 0.8667 |
| Support Vector Machine (SVM) | 86.65% | 82.11% | 43.98% | 57.28% | 0.8644 |
| **XGBoost (Selected)** | **86.65%** | **75.55%** | **50.86%** | **60.79%** | **0.8840** |

### Fine-Tuning Summary
- **Selected Model**: XGBoost
- **Optimal Hyperparameters (Grid Search)**: `{'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}`
- **GridSearch Cross-Validation ROC-AUC**: **85.80%**
- **Test Set ROC-AUC**: **88.18%** (Deployed Version v1)

---

## 6. Dashboard Visuals

Here are screenshots capturing the interactive dashboard platform:

### Executive Analytics Homepage
![Executive Overview Dashboard](file:///C:/Users/admin/.gemini/antigravity-ide/brain/0317a5ca-d3ea-4a34-ba0b-f8dcc671d447/dashboard_initial_view_1782314087675.png)

### Real-Time Client Risk Prediction
![Interactive Churn Predictor Form](file:///C:/Users/admin/.gemini/antigravity-ide/brain/0317a5ca-d3ea-4a34-ba0b-f8dcc671d447/prediction_result_view_1782314156859.png)
