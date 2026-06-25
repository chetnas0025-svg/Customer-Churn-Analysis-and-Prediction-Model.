import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Handle XGBoost import with GradientBoosting fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def main():
    print("Starting Model Training & Comparison...")
    
    # 1. Load data
    temp_dir = os.path.join('ml', 'temp_data')
    X_train = pd.read_csv(os.path.join(temp_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(temp_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(temp_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(temp_dir, 'y_test.csv')).values.ravel()
    
    # Load preprocessor
    preprocessor_path = os.path.join('ml', 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run preprocessing script first.")
    
    preprocessor = joblib.load(preprocessor_path)
    
    # 2. Preprocess features
    print("Preprocessing training and test features...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get list of feature names after preprocessing (useful for reference)
    # We can reconstruct column names if we want, but for training, numpy arrays are fine
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42) # probability=True is required to get predict_proba for ROC-AUC
    }
    
    if HAS_XGBOOST:
        print("XGBoost is available. Adding XGBoost to training list.")
        models['XGBoost'] = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=1.5,
                                          random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    else:
        print("XGBoost not found. Adding Gradient Boosting Classifier as fallback.")
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                                 learning_rate=0.1, random_state=42)
        
    metrics_list = []
    
    # Set up plotting for ROC Curves
    plt.figure(figsize=(10, 8))
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"{model_name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        metrics_list.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    comparison_path = os.path.join('ml', 'model_comparison.csv')
    metrics_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison table to {comparison_path}")
    
    # Save comparison to powerbi folder too (requirement of Phase 4)
    powerbi_dir = 'powerbi'
    os.makedirs(powerbi_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(powerbi_dir, 'model_metrics.csv'), index=False)
    print(f"Saved copy of model comparison table to {os.path.join(powerbi_dir, 'model_metrics.csv')}")
    
    # Finalize and save ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    charts_dir = os.path.join('ml', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'roc_comparison.png'), dpi=300)
    plt.close()
    
    print("ROC curve comparison plot saved to /ml/charts/roc_comparison.png")

if __name__ == '__main__':
    main()
