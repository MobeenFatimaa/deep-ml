SELECT 
    plan,
    DATE_TRUNC('month', signup_date) AS cohort_month,
    COUNT(user_id) AS cohort_size,
    ROUND(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '1 month' THEN 1 ELSE 0 END)::numeric / COUNT(user_id), 4) AS retention_m1,
    ROUND(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '2 month' THEN 1 ELSE 0 END)::numeric / COUNT(user_id), 4) AS retention_m2,
    ROUND(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '3 month' THEN 1 ELSE 0 END)::numeric / COUNT(user_id), 4) AS retention_m3
FROM subscriptions
GROUP BY plan, DATE_TRUNC('month', signup_date)
ORDER BY plan, cohort_month;