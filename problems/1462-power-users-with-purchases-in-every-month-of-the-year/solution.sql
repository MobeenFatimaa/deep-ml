SELECT 
    user_id
FROM (
    SELECT 
        user_id,
        EXTRACT(MONTH FROM purchase_date) AS purchase_month
    FROM purchases
    WHERE purchase_date >= '2024-01-01' 
      AND purchase_date < '2025-01-01'
    GROUP BY 
        user_id, 
        EXTRACT(MONTH FROM purchase_date)
    HAVING COUNT(purchase_id) >= 2
) AS qualifying_months
GROUP BY user_id
HAVING COUNT(DISTINCT purchase_month) = 12
ORDER BY user_id ASC;