import random
import math
import sqlite3
import os

def generate_dataset():
    random.seed(42)
    
    # Ensure directory structure exists
    os.makedirs('sql', exist_ok=True)
    
    customers = []
    base_id = 15600000 # Realistic bank customer ID prefix
    
    for i in range(10000):
        customer_id = base_id + i
        credit_score = int(max(350, min(850, random.gauss(650, 95))))
        geography = random.choices(['Maharashtra', 'Karnataka', 'Delhi'], weights=[0.50, 0.25, 0.25])[0]
        gender = random.choices(['Male', 'Female'], weights=[0.54, 0.46])[0]
        
        # Age distribution: centered around 39, standard deviation 10
        age = int(max(18, min(92, random.gauss(39, 10.5))))
        
        tenure = random.randint(0, 10)
        
        # Balance distribution
        if geography == 'Karnataka':
            # Karnataka has very few zero balances
            has_zero_balance = random.random() < 0.05
        else:
            has_zero_balance = random.random() < 0.40
            
        if has_zero_balance:
            balance = 0.0
        else:
            balance = round(max(10000.0, random.gauss(120000, 32000)) * 100, 2)
            
        # Products distribution
        num_products = random.choices([1, 2, 3, 4], weights=[0.51, 0.44, 0.04, 0.01])[0]
        
        has_credit_card = random.choices([1, 0], weights=[0.71, 0.29])[0]
        is_active_member = random.choices([1, 0], weights=[0.51, 0.49])[0]
        
        estimated_salary = round(random.uniform(10000.0, 200000.0) * 100, 2)
        
        # Determine exit probability using a logit model for realistic correlations
        z = -2.3
        
        # Age: older people churn more
        z += 0.075 * (age - 37)
        
        # Geography: Karnataka (Germany proxy) churns much more, Delhi slightly more than Maharashtra
        if geography == 'Karnataka':
            z += 0.85
        elif geography == 'Delhi':
            z += 0.05
            
        # Gender: Females churn slightly more
        if gender == 'Female':
            z += 0.3
            
        # Active status: active members are far less likely to churn
        if is_active_member == 1:
            z -= 1.0
            
        # Products: 2 products is highly stable; 1 is normal; 3 or 4 have extreme churn
        if num_products == 1:
            z += 0.25
        elif num_products == 2:
            z -= 1.2
        elif num_products == 3:
            z += 1.85
        elif num_products == 4:
            z += 2.95
            
        # Balance: higher balances are slightly more prone to churn, especially over 100k
        if balance > 0:
            z += 0.15 * ((balance / 100.0) / 100000.0)
            
        # Credit score: low credit scores (< 400) have very high churn risk
        if credit_score < 400:
            z += 1.5
        elif credit_score < 500:
            z += 0.4
        elif credit_score < 600:
            z += 0.1
        else:
            # High credit score reduces churn slightly
            z -= 0.1 * ((credit_score - 600) / 100.0)
            
        # Salary: very minor positive effect
        z += 0.03 * ((estimated_salary / 100.0) / 100000.0)
        
        # Logistic activation
        prob = 1.0 / (1.0 + math.exp(-z))
        exited = 1 if random.random() < prob else 0
        
        customers.append((
            customer_id, credit_score, geography, gender, age, tenure,
            balance, num_products, has_credit_card, is_active_member,
            estimated_salary, exited
        ))
        
    return customers

def write_seed_file(customers, seed_path):
    print(f"Writing seed SQL statements to {seed_path}...")
    with open(seed_path, 'w', encoding='utf-8') as f:
        f.write("-- SQL Seed Data for Bank Customer Churn\n")
        f.write("BEGIN TRANSACTION;\n")
        for c in customers:
            # Escape strings if needed (none contain special characters here, but good practice)
            # Row mapping: customer_id, credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, exited
            f.write(f"INSERT INTO customers (customer_id, credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, exited) VALUES "
                    f"({c[0]}, {c[1]}, '{c[2]}', '{c[3]}', {c[4]}, {c[5]}, {c[6]}, {c[7]}, {c[8]}, {c[9]}, {c[10]}, {c[11]});\n")
        f.write("COMMIT;\n")

def populate_sqlite(schema_path, seed_path, db_path):
    print(f"Creating database and populating it from schema and seed data...")
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    
    # Read and execute seed data
    with open(seed_path, 'r', encoding='utf-8') as f:
        seed_sql = f.read()
    cursor.executescript(seed_sql)
    
    conn.commit()
    
    # Verify count
    cursor.execute("SELECT COUNT(*) FROM customers;")
    count = cursor.fetchone()[0]
    print(f"Database successfully populated with {count} records.")
    
    # Print some churn statistics
    cursor.execute("SELECT exited, COUNT(*), ROUND(COUNT(*) * 100.0 / ?, 2) FROM customers GROUP BY exited;", (count,))
    stats = cursor.fetchall()
    print("Churn counts:")
    for row in stats:
        print(f"  Exited={row[0]}: {row[1]} ({row[2]}%)")
        
    conn.close()

if __name__ == '__main__':
    customers = generate_dataset()
    seed_path = 'sql/seed_data.sql'
    schema_path = 'sql/schema.sql'
    db_path = 'bank_churn.db'
    
    write_seed_file(customers, seed_path)
    populate_sqlite(schema_path, seed_path, db_path)
