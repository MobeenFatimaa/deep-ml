SELECT 
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS total,
    ROUND(
        100.0 * (SUM(amount) - LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', sale_date))) 
        / LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', sale_date)), 
        2
    ) AS pct_change
FROM sales
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month ASC;