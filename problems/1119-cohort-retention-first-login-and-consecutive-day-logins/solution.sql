-- your query
SELECT
user_id, MIN(login_date) AS first_login
FROM logins
WHERE user_id IN (
SELECT user_id
FROM (
SELECT user_id, login_date,
LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS next_login
FROM logins
)
WHERE next_login = login_date + INTERVAL '1 day'
)
GROUP BY user_id
ORDER BY user_id ASC;