import sqlite3
import csv
import os
import shutil

def export_powerbi_files():
    db_path = 'bank_churn.db'
    pbi_dir = 'powerbi'
    os.makedirs(pbi_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Master Customer dataset: churn_master.csv
    print("Exporting churn_master.csv...")
    cursor.execute("SELECT * FROM customers;")
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with open(os.path.join(pbi_dir, 'churn_master.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    # 2. Geography summary: geography_summary.csv
    print("Exporting geography_summary.csv...")
    cursor.execute("""
        SELECT 
            geography,
            COUNT(*) AS total_customers,
            SUM(exited) AS churned_customers,
            ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY geography;
    """)
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with open(os.path.join(pbi_dir, 'geography_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    # 3. Age summary: age_summary.csv
    print("Exporting age_summary.csv...")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN age < 30 THEN '<30'
                WHEN age >= 30 AND age < 45 THEN '30-45'
                WHEN age >= 45 AND age < 60 THEN '45-60'
                ELSE '60+'
            END AS age_group,
            COUNT(*) AS total_customers,
            SUM(exited) AS churned_customers,
            ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY age_group;
    """)
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with open(os.path.join(pbi_dir, 'age_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    # 4. Product summary: product_summary.csv
    print("Exporting product_summary.csv...")
    cursor.execute("""
        SELECT 
            num_products,
            COUNT(*) AS total_customers,
            SUM(exited) AS churned_customers,
            ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY num_products;
    """)
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with open(os.path.join(pbi_dir, 'product_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    # 5. Credit score summary: credit_score_summary.csv
    print("Exporting credit_score_summary.csv...")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN credit_score < 400 THEN '<400'
                WHEN credit_score >= 400 AND credit_score < 500 THEN '400-500'
                WHEN credit_score >= 500 AND credit_score < 600 THEN '500-600'
                WHEN credit_score >= 600 AND credit_score < 700 THEN '600-700'
                WHEN credit_score >= 700 AND credit_score < 800 THEN '700-800'
                ELSE '800+'
            END AS credit_score_range,
            COUNT(*) AS total_customers,
            SUM(exited) AS churned_customers,
            ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY credit_score_range;
    """)
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with open(os.path.join(pbi_dir, 'credit_score_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    conn.close()
    
    # 6. Copy model comparison table if it exists
    metrics_src = os.path.join('ml', 'model_comparison.csv')
    metrics_dst = os.path.join(pbi_dir, 'model_metrics.csv')
    if os.path.exists(metrics_src):
        print(f"Copying {metrics_src} to {metrics_dst}...")
        shutil.copy(metrics_src, metrics_dst)
    else:
        print("Model comparison CSV not generated yet.")
        
    print("Power BI summary exports completed successfully!")

if __name__ == '__main__':
    export_powerbi_files()
