import sqlite3
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

def main():
    print("Starting Data Preprocessing...")
    
    # 1. Load data from SQLite
    db_path = 'bank_churn.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found. Please run Phase 1 first.")
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    
    # Separate features and target
    X = df.drop(columns=['customer_id', 'exited'])
    y = df['exited']
    
    # Apply Feature Engineering
    from feature_engineering import add_engineered_features
    X = add_engineered_features(X)
    
    # Identify feature types
    num_cols = ['credit_score', 'age', 'tenure', 'balance', 'num_products', 'estimated_salary', 'balance_to_salary_ratio', 'balance_per_product']
    cat_cols = ['geography', 'gender']
    bin_cols = ['has_credit_card', 'is_active_member', 'is_high_product_risk', 'is_inactive_and_older', 'is_active_and_young']
    
    # 2. Handle missing values, scaling and encoding via ColumnTransformer
    # We include imputers to handle any potential missing values robustly
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # For binary columns, just make sure there are no missing values
    bin_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
            ('bin', bin_transformer, bin_cols)
        ],
        remainder='drop'
    )
    
    # 3. Train-test split (80/20) with stratification to handle imbalanced churn label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]} rows")
    print(f"Test size: {X_test.shape[0]} rows")
    
    # Fit the preprocessor on training data
    print("Fitting preprocessor...")
    preprocessor.fit(X_train)
    
    # Save the preprocessor
    preprocessor_path = os.path.join('ml', 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved successfully to {preprocessor_path}")
    
    # Save train/test datasets to temporary csvs for training models
    os.makedirs(os.path.join('ml', 'temp_data'), exist_ok=True)
    X_train.to_csv(os.path.join('ml', 'temp_data', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join('ml', 'temp_data', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join('ml', 'temp_data', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join('ml', 'temp_data', 'y_test.csv'), index=False)
    print("Temporary train/test splits saved to ml/temp_data/ for subsequent steps.")

if __name__ == '__main__':
    main()
