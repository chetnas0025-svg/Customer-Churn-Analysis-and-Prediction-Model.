import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

# Import possible models for reconstruction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_feature_names(preprocessor):
    """
    Dynamically extracts feature names from fitted ColumnTransformer
    """
    # 1. Num features
    num_cols = list(preprocessor.transformers_[0][2])
    
    # 2. Categorical features after OneHotEncoder
    onehot_step = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_cols_raw = preprocessor.transformers_[1][2]
    onehot_features = list(onehot_step.get_feature_names_out(cat_cols_raw))
    
    # 3. Binary columns passed through
    bin_cols = list(preprocessor.transformers_[2][2])
    
    return num_cols + onehot_features + bin_cols

def main():
    print("Selecting and tuning the Best Model...")
    
    # Load comparison results
    comparison_path = os.path.join('ml', 'model_comparison.csv')
    if not os.path.exists(comparison_path):
        raise FileNotFoundError(f"Comparison file {comparison_path} not found. Run model training first.")
        
    df_metrics = pd.read_csv(comparison_path)
    print("Model metrics comparison:")
    print(df_metrics)
    
    # Select best model based on ROC-AUC
    best_row = df_metrics.loc[df_metrics['ROC-AUC'].idxmax()]
    best_model_name = best_row['Model']
    best_auc = best_row['ROC-AUC']
    print(f"\nBest Model Selected: {best_model_name} (ROC-AUC: {best_auc})")
    
    # Load datasets
    temp_dir = os.path.join('ml', 'temp_data')
    X_train = pd.read_csv(os.path.join(temp_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(temp_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(temp_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(temp_dir, 'y_test.csv')).values.ravel()
    
    # Load preprocessor
    preprocessor = joblib.load(os.path.join('ml', 'preprocessor.pkl'))
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)
    
    # Setup grid search parameters depending on the selected model
    if best_model_name == 'Random Forest':
        model_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [8, 12, None],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == 'XGBoost':
        model_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [4, 5],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [1.0, 1.5, 2.0]
        }
    elif best_model_name == 'Gradient Boosting':
        model_base = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    elif best_model_name == 'SVM':
        model_base = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto']
        }
    elif best_model_name == 'Logistic Regression':
        model_base = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    else:
        raise ValueError(f"Unknown model name: {best_model_name}")
        
    print(f"Running GridSearchCV on {best_model_name} with parameters: {param_grid}")
    
    # Grid search cross-validation (using 5 folds, cv=5)
    grid_search = GridSearchCV(estimator=model_base, param_grid=param_grid, 
                               cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_processed, y_train)
    
    best_estimator = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate model on test set
    y_pred = best_estimator.predict(X_test_processed)
    y_prob = best_estimator.predict_proba(X_test_processed)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    
    # Print classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC on Test Set: {test_auc:.4f}")

    # Register the new model version in models/model_registry.csv
    import datetime
    import shutil
    
    registry_path = os.path.join('models', 'model_registry.csv')
    versions_dir = os.path.join('models', 'versions')
    os.makedirs(versions_dir, exist_ok=True)
    os.makedirs('ml', exist_ok=True)
    
    version = 'v1'
    if os.path.exists(registry_path):
        try:
            df_reg = pd.read_csv(registry_path)
            if not df_reg.empty:
                # Find maximum version index
                versions = df_reg['version'].str.replace('v', '').astype(int)
                next_version_num = versions.max() + 1
                version = f'v{next_version_num}'
        except Exception as e:
            print(f"Error reading registry, defaulting to v1: {e}")
            
    model_ver_path = os.path.join(versions_dir, f'model_{version}.pkl')
    preprocessor_ver_path = os.path.join(versions_dir, f'preprocessor_{version}.pkl')
    
    joblib.dump(best_estimator, model_ver_path)
    joblib.dump(preprocessor, preprocessor_ver_path)
    print(f"Saved model version to {model_ver_path}")
    print(f"Saved preprocessor version to {preprocessor_ver_path}")
    
    if os.path.exists(registry_path):
        try:
            df_reg = pd.read_csv(registry_path)
            df_reg['is_active'] = 0
        except Exception as e:
            df_reg = pd.DataFrame(columns=['version', 'training_date', 'dataset_size', 'roc_auc', 'model_path', 'preprocessor_path', 'is_active'])
    else:
        df_reg = pd.DataFrame(columns=['version', 'training_date', 'dataset_size', 'roc_auc', 'model_path', 'preprocessor_path', 'is_active'])
        
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {
        'version': version,
        'training_date': now_str,
        'dataset_size': int(len(X_train) + len(X_test)),
        'roc_auc': float(round(test_auc, 4)),
        'model_path': model_ver_path.replace('\\', '/'),
        'preprocessor_path': preprocessor_ver_path.replace('\\', '/'),
        'is_active': 1
    }
    
    df_reg = pd.concat([df_reg, pd.DataFrame([new_row])], ignore_index=True)
    df_reg.to_csv(registry_path, index=False)
    print(f"Model registry updated with version {version}")
    
    # Copy new active version to ml/best_model.pkl and ml/preprocessor.pkl
    shutil.copy(model_ver_path, os.path.join('ml', 'best_model.pkl'))
    shutil.copy(preprocessor_ver_path, os.path.join('ml', 'preprocessor.pkl'))
    print("Active version successfully copied to ml/best_model.pkl and ml/preprocessor.pkl")
    
    # ----------------------------------------------------
    # Generate Visualizations
    # ----------------------------------------------------
    charts_dir = os.path.join('ml', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. Feature Importance / Permutation Importance
    print("Generating Permutation Importance...")
    # Calculate permutation importance on test set
    result = permutation_importance(best_estimator, X_test_processed, y_test, 
                                    n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_importances_idx = result.importances_mean.argsort()[::-1]
    
    # Select top 10
    top_indices = sorted_importances_idx[:10]
    top_importances = result.importances_mean[top_indices]
    top_names = [feature_names[idx] for idx in top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_names, palette='mako')
    plt.title(f'Permutation Importance (Top 10 Features) - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Accuracy Drop', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print("Feature importance and confusion matrix saved to /ml/charts/")

if __name__ == '__main__':
    main()
