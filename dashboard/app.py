from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import sys

app = Flask(__name__)

# Add workspace and ML directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import importlib.util
predict_spec = importlib.util.spec_from_file_location(
    "predict_module", 
    os.path.join(os.path.dirname(__file__), '..', 'ml', '05_predict.py')
)
predict_module = importlib.util.module_from_spec(predict_spec)
sys.modules["predict_module"] = predict_module
predict_spec.loader.exec_module(predict_module)
predict_churn = predict_module.predict_churn
load_prediction_pipeline = predict_module.load_prediction_pipeline

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'bank_churn.db')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'preprocessor.pkl')
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'best_model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'model_comparison.csv')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    # 1. Fetch KPI cards and chart data from SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # KPIs
    cursor.execute("SELECT COUNT(*) as total, AVG(exited)*100 as churn_rate, AVG(balance) as avg_balance, AVG(credit_score) as avg_credit_score FROM customers;")
    kpi = cursor.fetchone()
    
    kpis = {
        'total_customers': f"{kpi['total']:,}",
        'churn_rate': f"{kpi['churn_rate']:.2f}%",
        'avg_balance': f"₹{kpi['avg_balance']:,.2f}",
        'avg_credit_score': f"{kpi['avg_credit_score']:.1f}"
    }
    
    # Churn by Geography
    cursor.execute("SELECT geography, COUNT(*) as count, SUM(exited) as churned, AVG(exited)*100 as rate FROM customers GROUP BY geography;")
    geo_rows = cursor.fetchall()
    geo_data = {
        'labels': [row['geography'] for row in geo_rows],
        'counts': [row['count'] for row in geo_rows],
        'churned': [row['churned'] for row in geo_rows],
        'rates': [round(row['rate'], 2) for row in geo_rows]
    }
    
    # Churn by Age Group
    cursor.execute("""
        SELECT 
            CASE 
                WHEN age < 30 THEN '<30'
                WHEN age >= 30 AND age < 45 THEN '30-45'
                WHEN age >= 45 AND age < 60 THEN '45-60'
                ELSE '60+'
            END AS age_group,
            COUNT(*) as count,
            SUM(exited) as churned
        FROM customers
        GROUP BY age_group
        ORDER BY 
            CASE age_group 
                WHEN '<30' THEN 1 
                WHEN '30-45' THEN 2 
                WHEN '45-60' THEN 3 
                ELSE 4 
            END;
    """)
    age_rows = cursor.fetchall()
    age_data = {
        'labels': [row['age_group'] for row in age_rows],
        'counts': [row['count'] for row in age_rows],
        'churned': [row['churned'] for row in age_rows],
        'retained': [row['count'] - row['churned'] for row in age_rows],
        'rates': [round((row['churned'] / row['count']) * 100, 2) for row in age_rows]
    }
    
    # Churn by Products
    cursor.execute("SELECT num_products, COUNT(*) as count, SUM(exited) as churned FROM customers GROUP BY num_products ORDER BY num_products;")
    prod_rows = cursor.fetchall()
    prod_data = {
        'labels': [str(row['num_products']) for row in prod_rows],
        'counts': [row['count'] for row in prod_rows],
        'churned': [row['churned'] for row in prod_rows],
        'retained': [row['count'] - row['churned'] for row in prod_rows],
        'rates': [round((row['churned'] / row['count']) * 100, 2) for row in prod_rows]
    }
    
    # Credit Score Distribution split by churn (Binned for Chart.js)
    cursor.execute("""
        SELECT 
            CASE 
                WHEN credit_score < 400 THEN '<400'
                WHEN credit_score >= 400 AND credit_score < 500 THEN '400-500'
                WHEN credit_score >= 500 AND credit_score < 600 THEN '500-600'
                WHEN credit_score >= 600 AND credit_score < 700 THEN '600-700'
                WHEN credit_score >= 700 AND credit_score < 800 THEN '700-800'
                ELSE '800+'
            END AS credit_range,
            exited,
            COUNT(*) as count
        FROM customers
        GROUP BY credit_range, exited
        ORDER BY 
            CASE credit_range
                WHEN '<400' THEN 1
                WHEN '400-500' THEN 2
                WHEN '500-600' THEN 3
                WHEN '600-700' THEN 4
                WHEN '700-800' THEN 5
                ELSE 6
            END;
    """)
    credit_rows = cursor.fetchall()
    # Align binned data
    bins = ['<400', '400-500', '500-600', '600-700', '700-800', '800+']
    credit_retained = {b: 0 for b in bins}
    credit_churned = {b: 0 for b in bins}
    for row in credit_rows:
        rng = row['credit_range']
        ex = row['exited']
        cnt = row['count']
        if ex == 0:
            credit_retained[rng] = cnt
        else:
            credit_churned[rng] = cnt
            
    credit_data = {
        'labels': bins,
        'retained': [credit_retained[b] for b in bins],
        'churned': [credit_churned[b] for b in bins]
    }
    
    conn.close()
    
    # 2. Fetch Model version details from registry
    registry_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_registry.csv')
    version_info = {
        'version': 'v1',
        'training_date': 'N/A',
        'dataset_size': '10,000',
        'roc_auc': '0.8818'
    }
    if os.path.exists(registry_path):
        try:
            df_reg = pd.read_csv(registry_path)
            active_row = df_reg[df_reg['is_active'] == 1]
            if not active_row.empty:
                version_info = {
                    'version': active_row.iloc[0]['version'],
                    'training_date': active_row.iloc[0]['training_date'],
                    'dataset_size': f"{int(active_row.iloc[0]['dataset_size']):,}",
                    'roc_auc': f"{active_row.iloc[0]['roc_auc']:.4f}"
                }
        except Exception as e:
            print(f"Error reading model registry: {e}")

    # 3. Model Drift Detection
    # Compare average prediction probability on the last 100 records against SQLite baseline churn rate
    drift = 0.0
    drift_warning = False
    recent_mean = 0.0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(exited) FROM customers;")
        baseline_rate = cursor.fetchone()[0] or 0.2037
        
        df_recent = pd.read_sql_query("SELECT * FROM customers ORDER BY customer_id DESC LIMIT 100;", conn)
        conn.close()
        
        if len(df_recent) >= 10:
            preprocessor, model = load_prediction_pipeline()
            X_recent = df_recent.drop(columns=['customer_id', 'exited'])
            from ml.feature_engineering import add_engineered_features
            X_recent = add_engineered_features(X_recent)
            X_rec_proc = preprocessor.transform(X_recent)
            recent_probs = model.predict_proba(X_rec_proc)[:, 1]
            recent_mean = float(np.mean(recent_probs))
            drift = abs(recent_mean - baseline_rate)
            if drift > 0.05:
                drift_warning = True
    except Exception as e:
        print(f"Drift calculation skipped: {e}")
        
    drift_pct = f"{drift * 100:.2f}%"
    
    # 4. Fetch Model Comparison Metrics from CSV
    models_metrics = []
    if os.path.exists(METRICS_PATH):
        try:
            df_m = pd.read_csv(METRICS_PATH)
            models_metrics = df_m.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading model comparison CSV: {e}")
            
    # Find the best model based on ROC-AUC
    best_model_name = "N/A"
    if models_metrics:
        best_model_name = max(models_metrics, key=lambda x: x.get('ROC-AUC', 0))['Model']
        
    return render_template(
        'index.html',
        kpis=kpis,
        geo_data=geo_data,
        age_data=age_data,
        prod_data=prod_data,
        credit_data=credit_data,
        models_metrics=models_metrics,
        best_model_name=best_model_name,
        version_info=version_info,
        drift_pct=drift_pct,
        drift_warning=drift_warning
    )

def log_prediction_request(customer_dict, prob, risk):
    import csv
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                credit_score INTEGER,
                geography TEXT,
                gender TEXT,
                age INTEGER,
                tenure INTEGER,
                balance REAL,
                num_products INTEGER,
                has_credit_card INTEGER,
                is_active_member INTEGER,
                estimated_salary REAL,
                churn_probability REAL,
                risk_level TEXT
            );
        """)
        cursor.execute("""
            INSERT INTO predictions (
                credit_score, geography, gender, age, tenure, balance, 
                num_products, has_credit_card, is_active_member, 
                estimated_salary, churn_probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            customer_dict['credit_score'], customer_dict['geography'], customer_dict['gender'],
            customer_dict['age'], customer_dict['tenure'], customer_dict['balance'],
            customer_dict['num_products'], customer_dict['has_credit_card'], customer_dict['is_active_member'],
            customer_dict['estimated_salary'], prob, risk
        ))
        conn.commit()
        
        # Export to CSV
        pbi_dir = os.path.join(os.path.dirname(__file__), '..', 'powerbi')
        os.makedirs(pbi_dir, exist_ok=True)
        csv_path = os.path.join(pbi_dir, 'prediction_history.csv')
        
        cursor.execute("SELECT * FROM predictions;")
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows([list(row) for row in rows])
            
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load prediction pipeline
        preprocessor, model = load_prediction_pipeline()
        
        # Get data from AJAX POST request
        data = request.json
        
        customer_dict = {
            'credit_score': int(data.get('credit_score')),
            'geography': data.get('geography'),
            'gender': data.get('gender'),
            'age': int(data.get('age')),
            'tenure': int(data.get('tenure')),
            'balance': float(data.get('balance')),
            'num_products': int(data.get('num_products')),
            'has_credit_card': int(data.get('has_credit_card')),
            'is_active_member': int(data.get('is_active_member')),
            'estimated_salary': float(data.get('estimated_salary'))
        }
        
        # Run prediction
        prob, risk = predict_churn(customer_dict, preprocessor, model)
        
        # Log request to SQLite and CSV
        log_prediction_request(customer_dict, prob, risk)
        
        # Add a custom color code for the badge UI
        color_class = "danger" if risk == "High" else "warning" if risk == "Medium" else "success"
        
        return jsonify({
            'success': True,
            'probability': prob,
            'risk_label': risk,
            'color_class': color_class
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "retrain_module", 
            os.path.join(os.path.dirname(__file__), '..', 'ml', '06_retrain.py')
        )
        retrain_module = importlib.util.module_from_spec(spec)
        sys.modules["retrain_module"] = retrain_module
        spec.loader.exec_module(retrain_module)
        
        success, msg = retrain_module.run_retrain()
        
        return jsonify({
            'success': True,
            'updated': success,
            'message': msg
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.csv'):
            dest_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_records')
            os.makedirs(dest_dir, exist_ok=True)
            file_path = os.path.join(dest_dir, file.filename)
            file.save(file_path)
            
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ingest_module", 
                os.path.join(os.path.dirname(__file__), '..', 'ml', '07_ingest_new_data.py')
            )
            ingest_module = importlib.util.module_from_spec(spec)
            sys.modules["ingest_module"] = ingest_module
            spec.loader.exec_module(ingest_module)
            
            rows_added, msg = ingest_module.ingest_batch_files()
            
            return jsonify({
                'success': True,
                'rows_added': rows_added,
                'message': msg
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type, must be CSV'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Start flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
