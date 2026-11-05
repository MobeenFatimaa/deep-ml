WITH filtered_visits AS (
    SELECT DISTINCT user_id, visit_date
    FROM visits
    WHERE visit_date <= CURRENT_DATE - INTERVAL '1 day'
),
numbered_visits AS (
    SELECT 
        user_id,
        visit_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY visit_date) AS rn
    FROM filtered_visits
),
gaps AS (
    SELECT 
        user_id,
        visit_date - rn * INTERVAL '1 day' AS grp
    FROM numbered_visits
),
streak_lengths AS (
    SELECT 
        user_id,
        COUNT(*) AS streak_length
    FROM gaps
    GROUP BY user_id, grp
),
max_streaks AS (
    SELECT 
        user_id,
        MAX(streak_length) AS longest_streak
    FROM streak_lengths
    GROUP BY user_id
)
SELECT 
    user_id,
    longest_streak
FROM max_streaks
ORDER BY longest_streak DESC, user_id ASC
LIMIT 100;