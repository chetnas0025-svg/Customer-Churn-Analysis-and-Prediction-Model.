import os
import urllib.request
import pandas as pd
import sqlite3

def download_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "Churn_Modelling.csv")
    
    print(f"Downloading raw dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, raw_path)
        print(f"Downloaded raw dataset successfully to {raw_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please ensure the CSV file is placed at data/raw/Churn_Modelling.csv manually.")
        if not os.path.exists(raw_path):
            raise e
            
    return raw_path

def validate_and_clean(raw_path):
    print("Validating and cleaning dataset...")
    df = pd.read_csv(raw_path)
    initial_rows = df.shape[0]
    
    # Apply Indian localization
    print("Mapping Geography to Indian states (France -> Maharashtra, Germany -> Karnataka, Spain -> Delhi)...")
    df['Geography'] = df['Geography'].map({'France': 'Maharashtra', 'Germany': 'Karnataka', 'Spain': 'Delhi'})
    
    # Scale Balance and EstimatedSalary to Indian Rupees
    print("Scaling Balance and Estimated Salary by 100 to represent Indian Rupees (INR)...")
    df['Balance'] = df['Balance'] * 100
    df['EstimatedSalary'] = df['EstimatedSalary'] * 100
    
    # 1. Check Missing Values
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts[missing_counts > 0])
    
    # Fill any missing values if present (just in case)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # 2. Check and Remove Duplicate Records by CustomerId
    duplicates = df[df.duplicated(subset=['CustomerId'], keep=False)]
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate customer records. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['CustomerId'], keep='first')
    else:
        print("No duplicate customer records found.")
        
    # 3. Check for Invalid Ages (Age must be between 18 and 100)
    invalid_age_mask = (df['Age'] < 18) | (df['Age'] > 100)
    invalid_age_count = invalid_age_mask.sum()
    if invalid_age_count > 0:
        print(f"Found {invalid_age_count} records with invalid ages (under 18 or over 100). Removing them.")
        df = df[~invalid_age_mask]
        
    # 4. Check for Invalid Balances (Balance must be >= 0)
    invalid_balance_mask = df['Balance'] < 0
    invalid_balance_count = invalid_balance_mask.sum()
    if invalid_balance_count > 0:
        print(f"Found {invalid_balance_count} records with negative balances. Removing them.")
        df = df[~invalid_balance_mask]
        
    # 5. Check for Invalid Credit Scores (Credit Score must be between 300 and 850)
    invalid_score_mask = (df['CreditScore'] < 300) | (df['CreditScore'] > 850)
    invalid_score_count = invalid_score_mask.sum()
    if invalid_score_count > 0:
        print(f"Found {invalid_score_count} records with invalid credit scores. Removing them.")
        df = df[~invalid_score_mask]
        
    final_rows = df.shape[0]
    print(f"Data cleaning complete. Retained {final_rows} of {initial_rows} records ({initial_rows - final_rows} removed).")
    
    # 6. Map columns to database schema names
    # Mapping raw columns to matches:
    # customer_id, credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, exited
    column_mapping = {
        'CustomerId': 'customer_id',
        'CreditScore': 'credit_score',
        'Geography': 'geography',
        'Gender': 'gender',
        'Age': 'age',
        'Tenure': 'tenure',
        'Balance': 'balance',
        'NumOfProducts': 'num_products',
        'HasCrCard': 'has_credit_card',
        'IsActiveMember': 'is_active_member',
        'EstimatedSalary': 'estimated_salary',
        'Exited': 'exited'
    }
    
    # Drop unused columns (RowNumber, Surname)
    cols_to_keep = [col for col in df.columns if col in column_mapping]
    df_clean = df[cols_to_keep].rename(columns=column_mapping)
    
    # Reorder columns to match database schema
    db_cols = ['customer_id', 'credit_score', 'geography', 'gender', 'age', 'tenure',
               'balance', 'num_products', 'has_credit_card', 'is_active_member',
               'estimated_salary', 'exited']
    df_clean = df_clean[db_cols]
    
    # Save cleaned master CSV
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    master_path = os.path.join(processed_dir, "churn_master.csv")
    df_clean.to_csv(master_path, index=False)
    print(f"Cleaned master dataset saved to {master_path}")
    
    # Save copy to powerbi directory too
    os.makedirs("powerbi", exist_ok=True)
    df_clean.to_csv(os.path.join("powerbi", "churn_master.csv"), index=False)
    print(f"Cleaned master dataset copied to powerbi/churn_master.csv")
    
    return df_clean

def save_to_database(df, db_path="bank_churn.db"):
    print(f"Saving cleaned dataset to SQLite database {db_path}...")
    
    # Read schema SQL
    schema_path = os.path.join("sql", "schema.sql")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Reset schema
    cursor.execute("DROP TABLE IF EXISTS customers;")
    cursor.executescript(schema_sql)
    conn.commit()
    
    # Write dataframe to SQL table
    df.to_sql("customers", conn, if_exists="append", index=False)
    
    # Verify count
    cursor.execute("SELECT COUNT(*) FROM customers;")
    count = cursor.fetchone()[0]
    print(f"Successfully populated 'customers' table with {count} rows.")
    
    # Print churn statistics
    cursor.execute("SELECT exited, COUNT(*), ROUND(COUNT(*) * 100.0 / ?, 2) FROM customers GROUP BY exited;", (count,))
    stats = cursor.fetchall()
    print("Database Churn counts:")
    for row in stats:
        print(f"  Exited={row[0]}: {row[1]} ({row[2]}%)")
        
    conn.close()

if __name__ == '__main__':
    raw_path = download_data()
    df_clean = validate_and_clean(raw_path)
    save_to_database(df_clean)
