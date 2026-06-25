-- Churn Rate by Geography
SELECT 
    geography,
    COUNT(*) AS total_customers,
    SUM(exited) AS churned_customers,
    ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY geography
ORDER BY churn_rate_pct DESC;

-- Churn Rate by Age Group
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
GROUP BY age_group
ORDER BY 
    CASE 
        WHEN age_group = '<30' THEN 1
        WHEN age_group = '30-45' THEN 2
        WHEN age_group = '45-60' THEN 3
        ELSE 4
    END;

-- Churn Rate by Number of Products
SELECT 
    num_products,
    COUNT(*) AS total_customers,
    SUM(exited) AS churned_customers,
    ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY num_products
ORDER BY num_products;

-- Average Balance of Churned vs Retained Customers
SELECT 
    exited,
    COUNT(*) AS total_customers,
    ROUND(AVG(balance), 2) AS avg_balance
FROM customers
GROUP BY exited;

-- Churn Rate by Credit Score Range
SELECT 
    CASE 
        WHEN credit_score < 400 THEN '<400 (Very Poor)'
        WHEN credit_score >= 400 AND credit_score < 500 THEN '400-500 (Poor)'
        WHEN credit_score >= 500 AND credit_score < 600 THEN '500-600 (Fair)'
        WHEN credit_score >= 600 AND credit_score < 700 THEN '600-700 (Good)'
        WHEN credit_score >= 700 AND credit_score < 800 THEN '700-800 (Very Good)'
        ELSE '800+ (Excellent)'
    END AS credit_score_range,
    COUNT(*) AS total_customers,
    SUM(exited) AS churned_customers,
    ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY credit_score_range
ORDER BY 
    CASE 
        WHEN credit_score_range LIKE '<400%' THEN 1
        WHEN credit_score_range LIKE '400-500%' THEN 2
        WHEN credit_score_range LIKE '500-600%' THEN 3
        WHEN credit_score_range LIKE '600-700%' THEN 4
        WHEN credit_score_range LIKE '700-800%' THEN 5
        ELSE 6
    END;

-- Active vs Inactive Member Churn Comparison
SELECT 
    is_active_member,
    COUNT(*) AS total_customers,
    SUM(exited) AS churned_customers,
    ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY is_active_member;

-- Churn Trend (using tenure as proxy for time in bank)
SELECT 
    tenure AS tenure_years,
    COUNT(*) AS total_customers,
    SUM(exited) AS churned_customers,
    ROUND(AVG(exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY tenure
ORDER BY tenure;
