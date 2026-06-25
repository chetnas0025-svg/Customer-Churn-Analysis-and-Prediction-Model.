import os
import pandas as pd
import sqlite3
import shutil
import datetime
import importlib.util
import sys

# Dynamic import of 06_retrain.py to trigger retraining programmatically
def trigger_retraining():
    print("Triggering retraining pipeline programmatically...")
    spec = importlib.util.spec_from_file_location(
        "retrain_module", 
        os.path.join(os.path.dirname(__file__), "06_retrain.py")
    )
    retrain_module = importlib.util.module_from_spec(spec)
    sys.modules["retrain_module"] = retrain_module
    spec.loader.exec_module(retrain_module)
    success, msg = retrain_module.run_retrain()
    print(f"Retraining complete: Success={success}, Status={msg}")
    return success, msg

def ingest_batch_files():
    new_records_dir = os.path.join("data", "new_records")
    archive_dir = os.path.join(new_records_dir, "archive")
    os.makedirs(new_records_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    db_path = "bank_churn.db"
    
    # 1. Scan directory for CSV files
    csv_files = [f for f in os.listdir(new_records_dir) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print("No new CSV files found in data/new_records/")
        return 0, "No new files"
        
    print(f"Discovered {len(csv_files)} new CSV file(s) for ingestion.")
    
    total_appended_rows = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get existing customer IDs to prevent duplicate inserts
    cursor.execute("SELECT customer_id FROM customers;")
    existing_ids = set([row[0] for row in cursor.fetchall()])
    
    column_mapping = {
        'customerid': 'customer_id',
        'customer_id': 'customer_id',
        'creditscore': 'credit_score',
        'credit_score': 'credit_score',
        'geography': 'geography',
        'gender': 'gender',
        'age': 'age',
        'tenure': 'tenure',
        'balance': 'balance',
        'numofproducts': 'num_products',
        'num_products': 'num_products',
        'hascrcard': 'has_credit_card',
        'has_credit_card': 'has_credit_card',
        'isactivemember': 'is_active_member',
        'is_active_member': 'is_active_member',
        'estimatedsalary': 'estimated_salary',
        'estimated_salary': 'estimated_salary',
        'exited': 'exited'
    }
    
    required_cols = ['customer_id', 'credit_score', 'geography', 'gender', 'age', 
                     'tenure', 'balance', 'num_products', 'has_credit_card', 
                     'is_active_member', 'estimated_salary', 'exited']
    
    for filename in csv_files:
        file_path = os.path.join(new_records_dir, filename)
        print(f"\nProcessing file: {filename}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Map columns case-insensitively
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns=column_mapping)
            
            # Check schema validation
            missing_schema_cols = [col for col in required_cols if col not in df.columns]
            if missing_schema_cols:
                print(f"Skipping {filename}: Missing columns {missing_schema_cols}")
                continue
                
            # Filter columns
            df_to_ingest = df[required_cols].copy()
            
            # Remove duplicate IDs within this CSV
            df_to_ingest = df_to_ingest.drop_duplicates(subset=['customer_id'])
            
            # Remove duplicates against existing SQLite IDs
            initial_count = len(df_to_ingest)
            df_to_ingest = df_to_ingest[~df_to_ingest['customer_id'].isin(existing_ids)]
            appended_count = len(df_to_ingest)
            
            print(f"File validation: {initial_count} unique rows. {appended_count} are new to SQLite (not duplicates).")
            
            if appended_count > 0:
                # Validate values ranges
                df_to_ingest = df_to_ingest[
                    (df_to_ingest['age'] >= 18) & (df_to_ingest['age'] <= 100) &
                    (df_to_ingest['balance'] >= 0) &
                    (df_to_ingest['credit_score'] >= 300) & (df_to_ingest['credit_score'] <= 850)
                ]
                valid_appended_count = len(df_to_ingest)
                print(f"Range validations filtered out {appended_count - valid_appended_count} rows. Writing {valid_appended_count} rows.")
                
                if valid_appended_count > 0:
                    df_to_ingest.to_sql("customers", conn, if_exists="append", index=False)
                    total_appended_rows += valid_appended_count
                    
            # Move file to archive directory with timestamp suffix
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"{base_name}_{timestamp}{ext}"
            archive_path = os.path.join(archive_dir, archive_filename)
            
            conn.commit() # Commit database inserts before archiving
            shutil.move(file_path, archive_path)
            print(f"Archived {filename} to {archive_path}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    conn.close()
    
    print(f"\nIngestion summary: Successfully ingested {total_appended_rows} records.")
    
    # 2. Trigger retraining if new records were appended
    if total_appended_rows > 0:
        print("New records detected in DB. Running model retraining pipeline...")
        success, msg = trigger_retraining()
        return total_appended_rows, f"Retraining triggered: {msg}"
    else:
        print("No new database inserts were committed. Retraining skipped.")
        return 0, "No new database inserts committed"

if __name__ == '__main__':
    ingest_batch_files()
