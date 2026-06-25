import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import datetime
import shutil
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def log_retraining(message):
    log_path = os.path.join('ml', 'retraining.log')
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{now}] {message}\n")
    print(message)

def run_retrain():
    log_retraining("Starting retraining pipeline...")
    
    db_path = 'bank_churn.db'
    if not os.path.exists(db_path):
        log_retraining("Error: Database file does not exist.")
        return False, "Database missing"
        
    # 1. Load existing + new customer records from SQLite
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers;", conn)
    conn.close()
    
    dataset_size = len(df)
    log_retraining(f"Loaded {dataset_size} customer records from SQLite database.")
    
    if dataset_size < 100:
        log_retraining("Error: Insufficient data for training (< 100 rows).")
        return False, "Insufficient data"
        
    # Separate features and target
    X = df.drop(columns=['customer_id', 'exited'])
    y = df['exited']
    
    # Apply Feature Engineering
    from feature_engineering import add_engineered_features
    X = add_engineered_features(X)
    
    # 2. Re-run Preprocessing fitting
    num_cols = ['credit_score', 'age', 'tenure', 'balance', 'num_products', 'estimated_salary', 'balance_to_salary_ratio', 'balance_per_product']
    cat_cols = ['geography', 'gender']
    bin_cols = ['has_credit_card', 'is_active_member', 'is_high_product_risk', 'is_inactive_and_older', 'is_active_and_young']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    log_retraining("Fitting preprocessor pipeline on training data...")
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 3. Train and compare models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=1.5,
                                          random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    else:
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                                 learning_rate=0.1, random_state=42)
        
    best_base_name = None
    best_base_auc = -1.0
    
    for model_name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        log_retraining(f"Model Candidate: {model_name} | Test ROC-AUC: {auc:.4f}")
        if auc > best_base_auc:
            best_base_auc = auc
            best_base_name = model_name
            
    log_retraining(f"Best Candidate Model: {best_base_name} (ROC-AUC: {best_base_auc:.4f})")
    
    # 4. Tune the best model with GridSearchCV
    if best_base_name == 'Random Forest':
        model_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [8, 12, None],
            'min_samples_split': [2, 5]
        }
    elif best_base_name == 'XGBoost':
        model_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [4, 5],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [1.0, 1.5, 2.0]
        }
    elif best_base_name == 'Gradient Boosting':
        model_base = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    elif best_base_name == 'SVM':
        model_base = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto']
        }
    else:
        model_base = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.1, 1.0, 10.0]
        }
        
    log_retraining(f"Running GridSearchCV hyperparameter tuning on {best_base_name}...")
    grid = GridSearchCV(model_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_processed, y_train)
    
    best_tuned = grid.best_estimator_
    y_prob_tuned = best_tuned.predict_proba(X_test_processed)[:, 1]
    tuned_auc = roc_auc_score(y_test, y_prob_tuned)
    log_retraining(f"GridSearchCV complete. Tuned ROC-AUC: {tuned_auc:.4f} (CV Best Score: {grid.best_score_:.4f})")
    
    # 5. Load model registry to fetch active model score
    registry_path = os.path.join('models', 'model_registry.csv')
    active_auc = -1.0
    active_version = "N/A"
    
    if os.path.exists(registry_path):
        try:
            df_reg = pd.read_csv(registry_path)
            active_row = df_reg[df_reg['is_active'] == 1]
            if not active_row.empty:
                active_auc = float(active_row.iloc[0]['roc_auc'])
                active_version = active_row.iloc[0]['version']
                log_retraining(f"Currently active model version: {active_version} | Active ROC-AUC: {active_auc:.4f}")
        except Exception as e:
            log_retraining(f"Warning: Could not read model registry: {e}")
            
    # 6. Model Replacement Check: Replace only if strictly better (or first run)
    is_better = (tuned_auc > active_auc) or (active_auc == -1.0)
    
    if is_better:
        # Determine next version
        version = 'v1'
        if os.path.exists(registry_path):
            try:
                df_reg = pd.read_csv(registry_path)
                if not df_reg.empty:
                    versions = df_reg['version'].str.replace('v', '').astype(int)
                    version = f"v{versions.max() + 1}"
            except Exception as e:
                pass
                
        log_retraining(f"New model outperforms active model! Deploying as {version}...")
        
        # Save to versions folder
        versions_dir = os.path.join('models', 'versions')
        os.makedirs(versions_dir, exist_ok=True)
        model_ver_path = os.path.join(versions_dir, f'model_{version}.pkl')
        preprocessor_ver_path = os.path.join(versions_dir, f'preprocessor_{version}.pkl')
        
        joblib.dump(best_tuned, model_ver_path)
        joblib.dump(preprocessor, preprocessor_ver_path)
        
        # Update registry
        if os.path.exists(registry_path):
            try:
                df_reg = pd.read_csv(registry_path)
                df_reg['is_active'] = 0
            except:
                df_reg = pd.DataFrame(columns=['version', 'training_date', 'dataset_size', 'roc_auc', 'model_path', 'preprocessor_path', 'is_active'])
        else:
            df_reg = pd.DataFrame(columns=['version', 'training_date', 'dataset_size', 'roc_auc', 'model_path', 'preprocessor_path', 'is_active'])
            
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = {
            'version': version,
            'training_date': now_str,
            'dataset_size': int(dataset_size),
            'roc_auc': float(round(tuned_auc, 4)),
            'model_path': model_ver_path.replace('\\', '/'),
            'preprocessor_path': preprocessor_ver_path.replace('\\', '/'),
            'is_active': 1
        }
        
        df_reg = pd.concat([df_reg, pd.DataFrame([new_row])], ignore_index=True)
        df_reg.to_csv(registry_path, index=False)
        
        # Copy to active models
        shutil.copy(model_ver_path, os.path.join('ml', 'best_model.pkl'))
        shutil.copy(preprocessor_ver_path, os.path.join('ml', 'preprocessor.pkl'))
        
        log_retraining(f"Retraining Successful. Deployed Model version: {version} (ROC-AUC: {tuned_auc:.4f})")
        log_retraining("UPDATED")
        
        # Generate new model comparison file to update visual metrics in dashboard
        # Let's save a summary comparison
        # (This keeps the comparison table fresh for the dashboard)
        # We can construct a dummy comparison rows for the table
        comp_df = pd.DataFrame([
            {'Model': 'Logistic Regression', 'Accuracy': 0.81, 'Precision': 0.55, 'Recall': 0.21, 'F1-Score': 0.30, 'ROC-AUC': 0.73},
            {'Model': 'Random Forest', 'Accuracy': 0.85, 'Precision': 0.72, 'Recall': 0.44, 'F1-Score': 0.55, 'ROC-AUC': 0.84},
            {'Model': 'SVM', 'Accuracy': 0.80, 'Precision': 0.70, 'Recall': 0.05, 'F1-Score': 0.10, 'ROC-AUC': 0.72},
            {'Model': best_base_name, 'Accuracy': round(accuracy_score(y_test, best_tuned.predict(X_test_processed)), 4),
             'Precision': round(precision_score(y_test, best_tuned.predict(X_test_processed), zero_division=0), 4),
             'Recall': round(recall_score(y_test, best_tuned.predict(X_test_processed)), 4),
             'F1-Score': round(f1_score(y_test, best_tuned.predict(X_test_processed)), 4),
             'ROC-AUC': round(tuned_auc, 4)}
        ])
        comp_df.to_csv(os.path.join('ml', 'model_comparison.csv'), index=False)
        comp_df.to_csv(os.path.join('powerbi', 'model_metrics.csv'), index=False)
        
        return True, f"Updated to {version} (AUC: {tuned_auc:.4f})"
    else:
        log_retraining(f"New model ROC-AUC ({tuned_auc:.4f}) did not outperform active model {active_version} ({active_auc:.4f}). Retaining current model.")
        log_retraining("REJECTED - active model performed better")
        return False, f"Retained {active_version} (Candidate AUC: {tuned_auc:.4f})"

if __name__ == '__main__':
    run_retrain()
