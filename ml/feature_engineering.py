import pandas as pd

def add_engineered_features(df):
    """
    Computes 5 engineered features to capture interactions between balance,
    product count, age, and member activity.
    """
    df = df.copy()
    
    # 1. Balance to Salary Ratio
    df['balance_to_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1.0)
    
    # 2. Balance per Product
    df['balance_per_product'] = df['balance'] / df['num_products']
    
    # 3. High product risk indicator (num_products >= 3)
    df['is_high_product_risk'] = (df['num_products'] >= 3).astype(int)
    
    # 4. Inactive and older customer risk (age >= 45 and active == 0)
    df['is_inactive_and_older'] = ((df['age'] >= 45) & (df['is_active_member'] == 0)).astype(int)
    
    # 5. Active and young customer stability (age < 35 and active == 1)
    df['is_active_and_young'] = ((df['age'] < 35) & (df['is_active_member'] == 1)).astype(int)
    
    return df
