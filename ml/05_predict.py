import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_prediction_pipeline():
    """
    Loads the fitted preprocessor and the best trained model
    """
    preprocessor_path = os.path.join('ml', 'preprocessor.pkl')
    best_model_path = os.path.join('ml', 'best_model.pkl')
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(best_model_path):
        raise FileNotFoundError("Model and/or preprocessor files are missing. Please run scripts 02 and 04 first.")
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(best_model_path)
    
    return preprocessor, model

def predict_churn(customer_dict, preprocessor, model):
    """
    Accepts a single customer record as a dictionary and returns:
    - Churn probability (%)
    - Risk label (Low/Medium/High)
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Ensure all required features are present
    required_cols = ['credit_score', 'geography', 'gender', 'age', 'tenure', 
                     'balance', 'num_products', 'has_credit_card', 'is_active_member', 
                     'estimated_salary']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: '{col}'")
            
    # Apply Feature Engineering
    from feature_engineering import add_engineered_features
    df = add_engineered_features(df)
            
    # Apply preprocessing
    df_processed = preprocessor.transform(df)
    
    # Predict Churn Probability
    probability = model.predict_proba(df_processed)[0, 1]
    prob_percentage = round(float(probability) * 100, 2)
    
    # Determine Risk Label
    if prob_percentage < 30.0:
        risk_label = "Low"
    elif prob_percentage <= 60.0:
        risk_label = "Medium"
    else:
        risk_label = "High"
        
    return prob_percentage, risk_label

def main():
    print("Loading ML components for prediction...")
    try:
        preprocessor, model = load_prediction_pipeline()
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return
        
    # Sample 1: Low Risk customer
    # Young, active member, 2 products (the highly stable option), no balance
    customer_low = {
        'credit_score': 720,
        'geography': 'Maharashtra',
        'gender': 'Male',
        'age': 28,
        'tenure': 4,
        'balance': 0.0,
        'num_products': 2,
        'has_credit_card': 1,
        'is_active_member': 1,
        'estimated_salary': 8500000.00
    }
    
    # Sample 2: Medium Risk customer
    # Inactive member, older, Delhi, 1 product, has a balance
    customer_med = {
        'credit_score': 580,
        'geography': 'Delhi',
        'gender': 'Female',
        'age': 44,
        'tenure': 2,
        'balance': 9500000.00,
        'num_products': 1,
        'has_credit_card': 1,
        'is_active_member': 0,
        'estimated_salary': 12000000.00
    }
    
    # Sample 3: High Risk customer
    # Karnataka, older, 3 products, inactive, high balance, poor credit score
    customer_high = {
        'credit_score': 410,
        'geography': 'Karnataka',
        'gender': 'Female',
        'age': 52,
        'tenure': 1,
        'balance': 14500000.00,
        'num_products': 3,
        'has_credit_card': 0,
        'is_active_member': 0,
        'estimated_salary': 17500000.00
    }
    
    samples = [
        ("Low Risk Persona", customer_low),
        ("Medium Risk Persona", customer_med),
        ("High Risk Persona", customer_high)
    ]
    
    print("\n--- SAMPLE PREDICTIONS ---")
    for name, customer in samples:
        prob, risk = predict_churn(customer, preprocessor, model)
        print(f"\nProfile: {name}")
        print(f"Details: Age={customer['age']}, Country={customer['geography']}, Products={customer['num_products']}, Active={customer['is_active_member']}")
        print(f"Result : Churn Probability = {prob}% | Churn Risk = {risk}")

if __name__ == '__main__':
    main()
