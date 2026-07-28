SELECT 1;
SELECT COUNT(*)
FROM stg_google_play_apps;
SELECT COUNT(*)
FROM stg_google_play_reviews;
DESCRIBE dim_apps;
DESCRIBE stg_google_play_apps;
SELECT VERSION();
USE Google_Play;
SET FOREIGN_KEY_CHECKS = 0;
DELETE FROM fact_user_reviews;
DELETE FROM dim_apps;
ALTER TABLE dim_apps AUTO_INCREMENT = 1;
SET FOREIGN_KEY_CHECKS = 1;
INSERT INTO dim_apps (
        app_name,
        category,
        rating,
        reviews_count,
        size_in_mb,
        size_raw,
        installs_count,
        price_usd,
        content_rating,
        genres,
        last_updated_date,
        current_version,
        min_android_ver
    ) WITH ranked_apps AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY app
                ORDER BY CAST(COALESCE(NULLIF(reviews, ''), '0') AS UNSIGNED) DESC,
                    STR_TO_DATE(last_updated, '%M %e, %Y') DESC,
                    raw_app_id DESC
            ) AS rn
        FROM stg_google_play_apps
        WHERE category <> '1.9'
    )
SELECT TRIM(app),
    category,
    CASE
        WHEN rating REGEXP '^[0-9]+(\\.[0-9]+)?$' THEN CAST(rating AS DECIMAL(3, 2))
        ELSE NULL
    END,
    CASE
        WHEN reviews REGEXP '^[0-9]+$' THEN CAST(reviews AS UNSIGNED)
        ELSE 0
    END,
    CASE
        WHEN LOWER(size) LIKE '%m' THEN CAST(REPLACE(LOWER(size), 'm', '') AS DECIMAL(8, 2))
        WHEN LOWER(size) LIKE '%k' THEN CAST(REPLACE(LOWER(size), 'k', '') AS DECIMAL(8, 2)) / 1024
        ELSE NULL
    END,
    size,
    CASE
        WHEN REPLACE(REPLACE(installs, '+', ''), ',', '') REGEXP '^[0-9]+$' THEN CAST(
            REPLACE(REPLACE(installs, '+', ''), ',', '') AS UNSIGNED
        )
        ELSE 0
    END,
    CASE
        WHEN REPLACE(price, '$', '') REGEXP '^[0-9]+(\\.[0-9]+)?$' THEN CAST(REPLACE(price, '$', '') AS DECIMAL(8, 2))
        ELSE 0.00
    END,
    content_rating,
    genres,
    STR_TO_DATE(last_updated, '%M %e, %Y'),
    current_ver,
    android_ver
FROM ranked_apps
WHERE rn = 1;
SELECT app_name,
    price_usd,
    is_paid
FROM dim_apps
LIMIT 10;
DESCRIBE fact_user_reviews;
DESCRIBE stg_google_play_reviews;
INSERT INTO fact_user_reviews (
        app_id,
        app_name,
        review_text,
        sentiment,
        sentiment_polarity,
        sentiment_subjectivity
    )
SELECT d.app_id,
    d.app_name,
    r.translated_review,
    r.sentiment,
    CASE
        WHEN r.sentiment_polarity REGEXP '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(r.sentiment_polarity AS DECIMAL(6, 5))
        ELSE NULL
    END,
    CASE
        WHEN r.sentiment_subjectivity REGEXP '^[0-9]+(\\.[0-9]+)?$' THEN CAST(r.sentiment_subjectivity AS DECIMAL(6, 5))
        ELSE NULL
    END
FROM stg_google_play_reviews r
    INNER JOIN dim_apps d ON TRIM(r.app) = d.app_name;