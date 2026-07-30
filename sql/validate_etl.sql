-- ============================================================================
-- Google Play Store Analytics – ETL Data Quality Validation
-- ============================================================================
-- Purpose  : Read-only validation of data quality across staging, dimension,
--            and fact tables after the ETL pipeline has completed.
-- Database : Google_Play
-- ============================================================================

USE Google_Play;

-- ############################################################################
-- SECTION 1 : ROW COUNTS – STAGING TABLES
-- ############################################################################

-- 1.1  Total rows loaded into the apps staging table
SELECT 'stg_google_play_apps' AS table_name,
    COUNT(*)                  AS total_rows
FROM stg_google_play_apps;

-- 1.2  Total rows loaded into the reviews staging table
SELECT 'stg_google_play_reviews' AS table_name,
    COUNT(*)                     AS total_rows
FROM stg_google_play_reviews;

-- ############################################################################
-- SECTION 2 : ROW COUNTS – DIMENSION & FACT TABLES
-- ############################################################################

-- 2.1  Total rows in the apps dimension table
SELECT 'dim_apps' AS table_name,
    COUNT(*)      AS total_rows
FROM dim_apps;

-- 2.2  Total rows in the user reviews fact table
SELECT 'fact_user_reviews' AS table_name,
    COUNT(*)               AS total_rows
FROM fact_user_reviews;

-- ############################################################################
-- SECTION 3 : NULL CHECKS – CRITICAL COLUMNS
-- ############################################################################

-- 3.1  Apps with NULL app_name (should be zero)
SELECT 'dim_apps.app_name IS NULL' AS validation,
    COUNT(*)                       AS violation_count
FROM dim_apps
WHERE app_name IS NULL;

-- 3.2  Apps with NULL category
SELECT 'dim_apps.category IS NULL' AS validation,
    COUNT(*)                       AS violation_count
FROM dim_apps
WHERE category IS NULL;

-- 3.3  Apps with NULL rating (NULLs may be expected for unrated apps)
SELECT 'dim_apps.rating IS NULL' AS validation,
    COUNT(*)                     AS violation_count
FROM dim_apps
WHERE rating IS NULL;

-- 3.4  Apps with NULL installs_count
SELECT 'dim_apps.installs_count IS NULL' AS validation,
    COUNT(*)                             AS violation_count
FROM dim_apps
WHERE installs_count IS NULL;

-- 3.5  Apps with NULL content_rating
SELECT 'dim_apps.content_rating IS NULL' AS validation,
    COUNT(*)                             AS violation_count
FROM dim_apps
WHERE content_rating IS NULL;

-- 3.6  Reviews with NULL app_id (foreign key must not be NULL)
SELECT 'fact_user_reviews.app_id IS NULL' AS validation,
    COUNT(*)                              AS violation_count
FROM fact_user_reviews
WHERE app_id IS NULL;

-- 3.7  Reviews with NULL sentiment
SELECT 'fact_user_reviews.sentiment IS NULL' AS validation,
    COUNT(*)                                 AS violation_count
FROM fact_user_reviews
WHERE sentiment IS NULL;

-- ############################################################################
-- SECTION 4 : DUPLICATE DETECTION
-- ############################################################################

-- 4.1  Duplicate apps – apps sharing the same app_name in dim_apps
SELECT app_name,
    COUNT(*) AS occurrence_count
FROM dim_apps
GROUP BY app_name
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;

-- 4.2  Duplicate reviews – identical app_id + review_text combinations
SELECT app_id,
    review_text,
    COUNT(*) AS occurrence_count
FROM fact_user_reviews
WHERE review_text IS NOT NULL
GROUP BY app_id, review_text
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC
LIMIT 20;

-- ############################################################################
-- SECTION 5 : RANGE & DOMAIN VALIDATION
-- ############################################################################

-- 5.1  Ratings outside the valid range of 0 to 5
SELECT 'rating NOT BETWEEN 0 AND 5' AS validation,
    COUNT(*)                         AS violation_count
FROM dim_apps
WHERE rating IS NOT NULL
    AND (rating < 0 OR rating > 5);

-- 5.2  Negative install counts (installs should never be negative)
SELECT 'installs_count < 0' AS validation,
    COUNT(*)                AS violation_count
FROM dim_apps
WHERE installs_count < 0;

-- 5.3  Negative prices (price should be zero or positive)
SELECT 'price_usd < 0' AS validation,
    COUNT(*)            AS violation_count
FROM dim_apps
WHERE price_usd < 0;

-- 5.4  Invalid sentiment values (expected: Positive, Negative, Neutral)
SELECT sentiment,
    COUNT(*) AS total_rows
FROM fact_user_reviews
WHERE sentiment IS NOT NULL
    AND sentiment NOT IN ('Positive', 'Negative', 'Neutral')
GROUP BY sentiment
ORDER BY total_rows DESC;

-- 5.5  Sentiment polarity outside the valid range of -1 to 1
SELECT 'sentiment_polarity NOT BETWEEN -1 AND 1' AS validation,
    COUNT(*)                                      AS violation_count
FROM fact_user_reviews
WHERE sentiment_polarity IS NOT NULL
    AND (sentiment_polarity < -1 OR sentiment_polarity > 1);

-- 5.6  Sentiment subjectivity outside the valid range of 0 to 1
SELECT 'sentiment_subjectivity NOT BETWEEN 0 AND 1' AS validation,
    COUNT(*)                                         AS violation_count
FROM fact_user_reviews
WHERE sentiment_subjectivity IS NOT NULL
    AND (sentiment_subjectivity < 0 OR sentiment_subjectivity > 1);

-- ############################################################################
-- SECTION 6 : EMPTY / MISSING REVIEW TEXT
-- ############################################################################

-- 6.1  Reviews with NULL or empty review text
SELECT 'empty_or_null_review_text' AS validation,
    COUNT(*)                       AS violation_count
FROM fact_user_reviews
WHERE review_text IS NULL
    OR TRIM(review_text) = '';

-- ############################################################################
-- SECTION 7 : REFERENTIAL INTEGRITY – FOREIGN KEY VALIDATION
-- ############################################################################

-- 7.1  Orphan fact records – reviews referencing a non-existent app_id
SELECT 'orphan_fact_user_reviews' AS validation,
    COUNT(*)                      AS violation_count
FROM fact_user_reviews f
    LEFT JOIN dim_apps d ON f.app_id = d.app_id
WHERE d.app_id IS NULL;

-- ############################################################################
-- SECTION 8 : COVERAGE CHECKS
-- ############################################################################

-- 8.1  Apps in dim_apps that have zero reviews in fact_user_reviews
SELECT d.app_id,
    d.app_name,
    d.category
FROM dim_apps d
    LEFT JOIN fact_user_reviews f ON d.app_id = f.app_id
WHERE f.review_id IS NULL
ORDER BY d.app_name
LIMIT 25;

-- 8.2  Count of apps without any reviews
SELECT 'apps_without_reviews' AS validation,
    COUNT(*)                  AS total_count
FROM dim_apps d
    LEFT JOIN fact_user_reviews f ON d.app_id = f.app_id
WHERE f.review_id IS NULL;

-- ############################################################################
-- SECTION 9 : CATEGORY & GENRE VALIDATION
-- ############################################################################

-- 9.1  Categories with NULL values
SELECT 'categories_with_null' AS validation,
    COUNT(*)                  AS violation_count
FROM dim_apps
WHERE category IS NULL
    OR TRIM(category) = '';

-- 9.2  Distinct category count
SELECT 'distinct_categories' AS metric,
    COUNT(DISTINCT category) AS total_count
FROM dim_apps;

-- 9.3  Distinct genre count
SELECT 'distinct_genres' AS metric,
    COUNT(DISTINCT genres) AS total_count
FROM dim_apps;

-- 9.4  Full category breakdown (for visual inspection)
SELECT category,
    COUNT(*) AS app_count
FROM dim_apps
GROUP BY category
ORDER BY app_count DESC;

-- ############################################################################
-- SECTION 10 : DATE CONVERSION VALIDATION
-- ############################################################################

-- 10.1  Apps where last_updated_date failed to parse (NULL after STR_TO_DATE)
SELECT 'last_updated_date IS NULL' AS validation,
    COUNT(*)                       AS violation_count
FROM dim_apps
WHERE last_updated_date IS NULL;

-- 10.2  Apps with last_updated_date in the future (potential data error)
SELECT 'last_updated_date > NOW()' AS validation,
    COUNT(*)                       AS violation_count
FROM dim_apps
WHERE last_updated_date > CURDATE();

-- 10.3  Date range of last_updated_date values
SELECT MIN(last_updated_date) AS earliest_update,
    MAX(last_updated_date)    AS latest_update
FROM dim_apps
WHERE last_updated_date IS NOT NULL;

-- ############################################################################
-- SECTION 11 : SIZE CONVERSION VALIDATION
-- ############################################################################

-- 11.1  Apps where size_in_mb is NULL (size could not be converted)
SELECT 'size_in_mb IS NULL' AS validation,
    COUNT(*)                AS violation_count
FROM dim_apps
WHERE size_in_mb IS NULL;

-- 11.2  Apps with implausibly large sizes (> 500 MB)
SELECT app_name,
    size_in_mb,
    size_raw
FROM dim_apps
WHERE size_in_mb > 500
ORDER BY size_in_mb DESC
LIMIT 10;

-- 11.3  Size conversion comparison (raw vs. converted, sample for inspection)
SELECT app_name,
    size_raw,
    size_in_mb
FROM dim_apps
WHERE size_in_mb IS NOT NULL
ORDER BY size_in_mb DESC
LIMIT 10;

-- ############################################################################
-- SECTION 12 : SUMMARY STATISTICS
-- ############################################################################

-- 12.1  Rating statistics across all rated apps
SELECT 'rating' AS metric,
    MIN(rating) AS min_value,
    MAX(rating) AS max_value,
    ROUND(AVG(rating), 2) AS avg_value,
    COUNT(*) AS non_null_count
FROM dim_apps
WHERE rating IS NOT NULL;

-- 12.2  Install count statistics
SELECT 'installs_count'       AS metric,
    MIN(installs_count)       AS min_value,
    MAX(installs_count)       AS max_value,
    ROUND(AVG(installs_count), 0) AS avg_value,
    COUNT(*)                  AS non_null_count
FROM dim_apps
WHERE installs_count IS NOT NULL;

-- 12.3  Price statistics across all apps
SELECT 'price_usd'          AS metric,
    MIN(price_usd)          AS min_value,
    MAX(price_usd)          AS max_value,
    ROUND(AVG(price_usd), 2) AS avg_value,
    COUNT(*)                AS non_null_count
FROM dim_apps
WHERE price_usd IS NOT NULL;

-- 12.4  Sentiment polarity statistics
SELECT 'sentiment_polarity'          AS metric,
    MIN(sentiment_polarity)          AS min_value,
    MAX(sentiment_polarity)          AS max_value,
    ROUND(AVG(sentiment_polarity), 4) AS avg_value,
    COUNT(*)                         AS non_null_count
FROM fact_user_reviews
WHERE sentiment_polarity IS NOT NULL;

-- 12.5  Sentiment subjectivity statistics
SELECT 'sentiment_subjectivity'          AS metric,
    MIN(sentiment_subjectivity)          AS min_value,
    MAX(sentiment_subjectivity)          AS max_value,
    ROUND(AVG(sentiment_subjectivity), 4) AS avg_value,
    COUNT(*)                             AS non_null_count
FROM fact_user_reviews
WHERE sentiment_subjectivity IS NOT NULL;

-- ============================================================================
-- END OF VALIDATION SCRIPT
-- ============================================================================
