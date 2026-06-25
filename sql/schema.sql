-- Schema for bank customer churn database
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    credit_score INTEGER,
    geography TEXT CHECK(geography IN ('Maharashtra', 'Karnataka', 'Delhi')),
    gender TEXT CHECK(gender IN ('Male', 'Female')),
    age INTEGER,
    tenure INTEGER,
    balance REAL,
    num_products INTEGER,
    has_credit_card INTEGER CHECK(has_credit_card IN (0, 1)),
    is_active_member INTEGER CHECK(is_active_member IN (0, 1)),
    estimated_salary REAL,
    exited INTEGER CHECK(exited IN (0, 1))
);
