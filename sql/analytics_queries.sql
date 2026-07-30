-- ============================================================================
-- Google Play Store Analytics – Business Intelligence Queries
-- ============================================================================
-- Database : Google_Play
-- Tables   : dim_apps, fact_user_reviews
-- Purpose  : Analytical queries covering app performance,
--            ratings, categories, pricing, installs, reviews, sentiment,
--            and executive-level KPIs.
-- ============================================================================

USE Google_Play;

-- ############################################################################
-- SECTION 1 : APP PERFORMANCE
-- ############################################################################

-- Q01 | Which are the top 15 highest-performing apps by install volume?
-- Business value: Identifies market leaders to benchmark against.
SELECT app_name,
    category,
    installs_count,
    rating,
    reviews_count,
    RANK() OVER (ORDER BY installs_count DESC) AS install_rank
FROM dim_apps
WHERE installs_count IS NOT NULL
ORDER BY installs_count DESC,
    rating DESC
LIMIT 15;

-- Q02 | Which apps achieve the best rating among those with ≥ 1 M installs?
-- Business value: Surfaces high-quality apps that also have mass adoption.
SELECT app_name,
    category,
    rating,
    installs_count,
    DENSE_RANK() OVER (ORDER BY rating DESC, installs_count DESC) AS quality_rank
FROM dim_apps
WHERE installs_count >= 1000000
    AND rating IS NOT NULL
ORDER BY quality_rank
LIMIT 15;

-- Q03 | How does each app rank within its own category by installs?
-- Business value: Reveals category-level competitive positioning.
WITH ranked AS (
    SELECT app_name,
        category,
        installs_count,
        rating,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY installs_count DESC
        ) AS category_rank
    FROM dim_apps
    WHERE installs_count IS NOT NULL
)
SELECT app_name,
    category,
    installs_count,
    rating,
    category_rank
FROM ranked
WHERE category_rank <= 3
ORDER BY category, category_rank;

-- Q04 | Which apps have been updated most recently and still maintain a high rating?
-- Business value: Identifies actively maintained, high-quality products.
SELECT app_name,
    category,
    rating,
    last_updated_date,
    DATEDIFF(CURDATE(), last_updated_date) AS days_since_update
FROM dim_apps
WHERE rating >= 4.0
    AND last_updated_date IS NOT NULL
ORDER BY last_updated_date DESC
LIMIT 15;

-- ############################################################################
-- SECTION 2 : RATING ANALYTICS
-- ############################################################################

-- Q05 | What is the overall distribution of app ratings?
-- Business value: Reveals whether the store skews toward high or low quality.
SELECT CASE
        WHEN rating >= 4.5 THEN '4.5 – 5.0  (Excellent)'
        WHEN rating >= 4.0 THEN '4.0 – 4.4  (Very Good)'
        WHEN rating >= 3.5 THEN '3.5 – 3.9  (Good)'
        WHEN rating >= 3.0 THEN '3.0 – 3.4  (Average)'
        WHEN rating >= 2.0 THEN '2.0 – 2.9  (Below Avg)'
        ELSE                    '0.0 – 1.9  (Poor)'
    END AS rating_tier,
    COUNT(*) AS app_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY rating_tier
ORDER BY MIN(rating) DESC;

-- Q06 | How do free and paid app ratings compare?
-- Business value: Determines if monetisation model correlates with quality.
SELECT CASE WHEN is_paid = 1 THEN 'Paid' ELSE 'Free' END AS pricing_model,
    COUNT(*)                       AS total_apps,
    ROUND(AVG(rating), 2)         AS avg_rating,
    ROUND(MIN(rating), 2)         AS min_rating,
    ROUND(MAX(rating), 2)         AS max_rating,
    ROUND(AVG(installs_count), 0) AS avg_installs
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY pricing_model;

-- Q07 | Which content ratings correlate with the highest user satisfaction?
-- Business value: Helps content strategy teams target the right audience tier.
SELECT content_rating,
    COUNT(*)               AS total_apps,
    ROUND(AVG(rating), 2)  AS avg_rating,
    ROUND(AVG(installs_count), 0) AS avg_installs,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_store
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY content_rating
ORDER BY avg_rating DESC;

-- Q08 | Which individual apps are statistical outliers—rated far above their category average?
-- Business value: Flags breakout apps that outperform category norms.
WITH category_stats AS (
    SELECT category,
        AVG(rating)    AS avg_rating,
        STDDEV(rating) AS std_rating
    FROM dim_apps
    WHERE rating IS NOT NULL
    GROUP BY category
)
SELECT d.app_name,
    d.category,
    d.rating,
    ROUND(cs.avg_rating, 2)  AS category_avg,
    ROUND(d.rating - cs.avg_rating, 2) AS delta_vs_avg
FROM dim_apps d
    JOIN category_stats cs ON d.category = cs.category
WHERE d.rating IS NOT NULL
    AND cs.std_rating > 0
    AND (d.rating - cs.avg_rating) > 1.5 * cs.std_rating
ORDER BY delta_vs_avg DESC
LIMIT 20;

-- ############################################################################
-- SECTION 3 : CATEGORY ANALYSIS
-- ############################################################################

-- Q09 | What is the average rating and total app count per category?
-- Business value: Benchmarks category health across the store.
SELECT category,
    COUNT(*)              AS total_apps,
    ROUND(AVG(rating), 2) AS avg_rating,
    SUM(installs_count)   AS total_installs,
    RANK() OVER (ORDER BY AVG(rating) DESC) AS rating_rank
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY category
ORDER BY avg_rating DESC;

-- Q10 | Which categories dominate the store by app volume?
-- Business value: Identifies oversaturated vs. under-served categories.
SELECT category,
    COUNT(*) AS total_apps,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_store,
    ROUND(AVG(rating), 2) AS avg_rating
FROM dim_apps
GROUP BY category
ORDER BY total_apps DESC;

-- Q11 | Which genres (sub-categories) have the highest average ratings?
-- Business value: Reveals niche opportunities with high user satisfaction.
SELECT genres,
    COUNT(*)              AS total_apps,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(installs_count), 0) AS avg_installs
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY genres
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC
LIMIT 20;

-- Q12 | How does average app size vary across categories?
-- Business value: Informs size-budget decisions for new app development.
SELECT category,
    COUNT(*) AS total_apps,
    ROUND(AVG(size_in_mb), 2) AS avg_size_mb,
    ROUND(MIN(size_in_mb), 2) AS min_size_mb,
    ROUND(MAX(size_in_mb), 2) AS max_size_mb
FROM dim_apps
WHERE size_in_mb IS NOT NULL
GROUP BY category
ORDER BY avg_size_mb DESC;

-- ############################################################################
-- SECTION 4 : PRICING ANALYSIS
-- ############################################################################

-- Q13 | What is the price distribution of paid apps across categories?
-- Business value: Guides pricing strategy for new product launches.
SELECT category,
    COUNT(*)                AS paid_app_count,
    ROUND(MIN(price_usd), 2) AS min_price,
    ROUND(AVG(price_usd), 2) AS avg_price,
    ROUND(MAX(price_usd), 2) AS max_price,
    ROUND(SUM(price_usd), 2) AS total_listed_price
FROM dim_apps
WHERE is_paid = 1
GROUP BY category
ORDER BY avg_price DESC;

-- Q14 | What are the most expensive apps on the store?
-- Business value: Identifies the premium pricing ceiling per category.
WITH priced AS (
    SELECT app_name,
        category,
        price_usd,
        rating,
        installs_count,
        RANK() OVER (ORDER BY price_usd DESC) AS price_rank
    FROM dim_apps
    WHERE is_paid = 1
)
SELECT app_name,
    category,
    price_usd,
    rating,
    installs_count,
    price_rank
FROM priced
WHERE price_rank <= 15
ORDER BY price_rank;

-- Q15 | Is there a correlation pattern between price tier and rating?
-- Business value: Determines whether higher-priced apps receive better ratings.
SELECT CASE
        WHEN price_usd = 0    THEN 'Free'
        WHEN price_usd <= 0.99 THEN '$0.01 – $0.99'
        WHEN price_usd <= 4.99 THEN '$1.00 – $4.99'
        WHEN price_usd <= 9.99 THEN '$5.00 – $9.99'
        ELSE                        '$10.00+'
    END AS price_tier,
    COUNT(*)              AS total_apps,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(installs_count), 0) AS avg_installs,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM dim_apps
WHERE rating IS NOT NULL
GROUP BY price_tier
ORDER BY MIN(price_usd);

-- Q16 | What share of each category's apps are paid?
-- Business value: Reveals monetisation density per vertical.
SELECT category,
    COUNT(*) AS total_apps,
    SUM(is_paid) AS paid_apps,
    COUNT(*) - SUM(is_paid) AS free_apps,
    ROUND(SUM(is_paid) * 100.0 / COUNT(*), 2) AS paid_pct
FROM dim_apps
GROUP BY category
ORDER BY paid_pct DESC;

-- ############################################################################
-- SECTION 5 : INSTALL ANALYSIS
-- ############################################################################

-- Q17 | What is the install-volume distribution across the store?
-- Business value: Shows how many apps reach meaningful scale.
SELECT CASE
        WHEN installs_count >= 1000000000 THEN '1 B+'
        WHEN installs_count >= 100000000  THEN '100 M – 999 M'
        WHEN installs_count >= 10000000   THEN '10 M – 99 M'
        WHEN installs_count >= 1000000    THEN '1 M – 9.9 M'
        WHEN installs_count >= 100000     THEN '100 K – 999 K'
        WHEN installs_count >= 10000      THEN '10 K – 99 K'
        ELSE                                   '< 10 K'
    END AS install_bucket,
    COUNT(*) AS app_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM dim_apps
WHERE installs_count IS NOT NULL
GROUP BY install_bucket
ORDER BY MIN(installs_count) DESC;

-- Q18 | Which categories generate the most cumulative installs?
-- Business value: Quantifies total addressable reach per vertical.
SELECT category,
    COUNT(*)              AS total_apps,
    SUM(installs_count)   AS total_installs,
    ROUND(AVG(installs_count), 0) AS avg_installs_per_app,
    ROUND(
        SUM(installs_count) * 100.0
        / SUM(SUM(installs_count)) OVER (),
        2
    ) AS pct_of_total_installs
FROM dim_apps
WHERE installs_count IS NOT NULL
GROUP BY category
ORDER BY total_installs DESC;

-- Q19 | Among highly installed apps (≥ 10 M), which have the lowest ratings?
-- Business value: Flags popular apps at risk of user churn due to poor quality.
SELECT app_name,
    category,
    installs_count,
    rating,
    reviews_count
FROM dim_apps
WHERE installs_count >= 10000000
    AND rating IS NOT NULL
ORDER BY rating ASC,
    installs_count DESC
LIMIT 15;

-- Q20 | What is the median install count per category?
-- Business value: Median resists outlier distortion better than the mean.
WITH ordered AS (
    SELECT category,
        installs_count,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY installs_count)     AS rn,
        COUNT(*)      OVER (PARTITION BY category)                            AS cnt
    FROM dim_apps
    WHERE installs_count IS NOT NULL
)
SELECT category,
    ROUND(AVG(installs_count), 0) AS median_installs
FROM ordered
WHERE rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2))
GROUP BY category
ORDER BY median_installs DESC;

-- ############################################################################
-- SECTION 6 : REVIEW ANALYSIS
-- ############################################################################

-- Q21 | Which apps receive the highest volume of user reviews?
-- Business value: High review volume signals strong user engagement.
SELECT f.app_name,
    d.category,
    COUNT(*)              AS total_reviews,
    ROUND(AVG(f.sentiment_polarity), 3) AS avg_polarity,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS review_rank
FROM fact_user_reviews f
    JOIN dim_apps d ON f.app_id = d.app_id
GROUP BY f.app_name, d.category
ORDER BY total_reviews DESC
LIMIT 15;

-- Q22 | What percentage of reviews contain actual text vs. empty entries?
-- Business value: Measures the quality of user feedback data.
SELECT CASE
        WHEN review_text IS NULL OR TRIM(review_text) = '' THEN 'Empty / NULL'
        ELSE 'Has Text'
    END AS review_quality,
    COUNT(*) AS review_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM fact_user_reviews
GROUP BY review_quality;

-- Q23 | Which categories generate the most reviews per app on average?
-- Business value: Identifies verticals with the most vocal user bases.
SELECT d.category,
    COUNT(DISTINCT d.app_id) AS apps_with_reviews,
    COUNT(f.review_id)       AS total_reviews,
    ROUND(COUNT(f.review_id) * 1.0 / COUNT(DISTINCT d.app_id), 1) AS reviews_per_app
FROM dim_apps d
    JOIN fact_user_reviews f ON d.app_id = f.app_id
GROUP BY d.category
ORDER BY reviews_per_app DESC;

-- Q24 | How many apps in the store have zero user reviews?
-- Business value: Quantifies the "long tail" of unreviewed apps.
SELECT 'Apps WITH reviews'  AS segment,
    COUNT(DISTINCT f.app_id) AS app_count
FROM fact_user_reviews f
UNION ALL
SELECT 'Apps WITHOUT reviews' AS segment,
    COUNT(*) AS app_count
FROM dim_apps d
    LEFT JOIN fact_user_reviews f ON d.app_id = f.app_id
WHERE f.review_id IS NULL;

-- ############################################################################
-- SECTION 7 : SENTIMENT ANALYSIS
-- ############################################################################

-- Q25 | What is the overall sentiment split across all reviews?
-- Business value: Establishes the store-wide sentiment baseline.
SELECT sentiment,
    COUNT(*) AS total_reviews,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM fact_user_reviews
WHERE sentiment IS NOT NULL
GROUP BY sentiment
ORDER BY total_reviews DESC;

-- Q26 | Which apps have the highest percentage of negative reviews?
-- Business value: Identifies products that may need urgent quality attention.
WITH app_sentiment AS (
    SELECT app_name,
        COUNT(*) AS total_reviews,
        SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_count
    FROM fact_user_reviews
    WHERE sentiment IS NOT NULL
    GROUP BY app_name
    HAVING COUNT(*) >= 20
)
SELECT app_name,
    total_reviews,
    negative_count,
    ROUND(negative_count * 100.0 / total_reviews, 2) AS negative_pct
FROM app_sentiment
ORDER BY negative_pct DESC
LIMIT 15;

-- Q27 | Which categories have the highest average sentiment polarity?
-- Business value: Benchmarks user happiness across verticals.
SELECT d.category,
    COUNT(f.review_id) AS total_reviews,
    ROUND(AVG(f.sentiment_polarity), 4)    AS avg_polarity,
    ROUND(AVG(f.sentiment_subjectivity), 4) AS avg_subjectivity
FROM fact_user_reviews f
    JOIN dim_apps d ON f.app_id = d.app_id
WHERE f.sentiment_polarity IS NOT NULL
GROUP BY d.category
ORDER BY avg_polarity DESC;

-- Q28 | What is the sentiment breakdown for each content rating tier?
-- Business value: Reveals whether audience maturity level affects satisfaction.
SELECT d.content_rating,
    f.sentiment,
    COUNT(*) AS review_count,
    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (PARTITION BY d.content_rating),
        2
    ) AS pct_within_tier
FROM fact_user_reviews f
    JOIN dim_apps d ON f.app_id = d.app_id
WHERE f.sentiment IS NOT NULL
GROUP BY d.content_rating, f.sentiment
ORDER BY d.content_rating, review_count DESC;

-- Q29 | Which apps show the widest gap between polarity and subjectivity?
-- Business value: Detects apps with highly emotional yet opinionated review bases.
WITH app_metrics AS (
    SELECT app_name,
        COUNT(*)                                AS total_reviews,
        ROUND(AVG(sentiment_polarity), 4)       AS avg_polarity,
        ROUND(AVG(sentiment_subjectivity), 4)   AS avg_subjectivity
    FROM fact_user_reviews
    WHERE sentiment_polarity IS NOT NULL
        AND sentiment_subjectivity IS NOT NULL
    GROUP BY app_name
    HAVING COUNT(*) >= 20
)
SELECT app_name,
    total_reviews,
    avg_polarity,
    avg_subjectivity,
    ROUND(ABS(avg_polarity - avg_subjectivity), 4) AS polarity_subjectivity_gap
FROM app_metrics
ORDER BY polarity_subjectivity_gap DESC
LIMIT 15;

-- ############################################################################
-- SECTION 8 : BUSINESS INTELLIGENCE
-- ############################################################################

-- Q30 | What is the year-over-year update activity trend?
-- Business value: Shows whether the store ecosystem is accelerating or slowing.
SELECT YEAR(last_updated_date) AS update_year,
    COUNT(*)                   AS apps_updated,
    ROUND(AVG(rating), 2)      AS avg_rating_that_year
FROM dim_apps
WHERE last_updated_date IS NOT NULL
GROUP BY update_year
ORDER BY update_year;

-- Q31 | Which categories have the highest "quality × reach" score?
-- Business value: Composite metric balancing user satisfaction with scale.
WITH category_score AS (
    SELECT category,
        COUNT(*)              AS total_apps,
        ROUND(AVG(rating), 2) AS avg_rating,
        SUM(installs_count)   AS total_installs,
        ROUND(AVG(rating) * LOG10(SUM(installs_count) + 1), 2) AS quality_reach_score
    FROM dim_apps
    WHERE rating IS NOT NULL
        AND installs_count IS NOT NULL
    GROUP BY category
)
SELECT category,
    total_apps,
    avg_rating,
    total_installs,
    quality_reach_score,
    RANK() OVER (ORDER BY quality_reach_score DESC) AS composite_rank
FROM category_score
ORDER BY composite_rank;

-- Q32 | How does sentiment polarity differ between free and paid apps?
-- Business value: Tests the hypothesis that paying users are more critical.
SELECT CASE WHEN d.is_paid = 1 THEN 'Paid' ELSE 'Free' END AS pricing_model,
    COUNT(f.review_id)                        AS total_reviews,
    ROUND(AVG(f.sentiment_polarity), 4)       AS avg_polarity,
    ROUND(AVG(f.sentiment_subjectivity), 4)   AS avg_subjectivity,
    SUM(CASE WHEN f.sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN f.sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_count,
    ROUND(
        SUM(CASE WHEN f.sentiment = 'Positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS positive_pct
FROM fact_user_reviews f
    JOIN dim_apps d ON f.app_id = d.app_id
WHERE f.sentiment IS NOT NULL
GROUP BY pricing_model;

-- Q33 | What is the competitive landscape per category — top 3 apps by a composite score?
-- Business value: Provides a category-level leaderboard combining multiple signals.
WITH scored AS (
    SELECT d.app_name,
        d.category,
        d.rating,
        d.installs_count,
        d.reviews_count,
        COALESCE(AVG(f.sentiment_polarity), 0) AS avg_polarity,
        ROUND(
            (COALESCE(d.rating, 0) / 5.0) * 0.4
            + (LOG10(d.installs_count + 1) / LOG10(MAX(d.installs_count) OVER () + 1)) * 0.3
            + (COALESCE(AVG(f.sentiment_polarity), 0) + 1) / 2.0 * 0.3,
            4
        ) AS composite_score
    FROM dim_apps d
        LEFT JOIN fact_user_reviews f ON d.app_id = f.app_id
    WHERE d.rating IS NOT NULL
        AND d.installs_count > 0
    GROUP BY d.app_id, d.app_name, d.category, d.rating,
             d.installs_count, d.reviews_count
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY composite_score DESC
        ) AS category_rank
    FROM scored
)
SELECT app_name,
    category,
    rating,
    installs_count,
    ROUND(avg_polarity, 3) AS avg_polarity,
    composite_score,
    category_rank
FROM ranked
WHERE category_rank <= 3
ORDER BY category, category_rank;

-- Q34 | Which Android version requirements are most common, and how do they affect ratings?
-- Business value: Informs minimum-SDK decisions for new projects.
SELECT min_android_ver,
    COUNT(*)              AS total_apps,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(installs_count), 0) AS avg_installs,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_store
FROM dim_apps
WHERE min_android_ver IS NOT NULL
    AND rating IS NOT NULL
GROUP BY min_android_ver
HAVING COUNT(*) >= 5
ORDER BY total_apps DESC
LIMIT 15;

-- ############################################################################
-- SECTION 9 : DASHBOARD KPIs
-- ############################################################################

-- Q35 | Executive KPI summary – single-row snapshot of the entire store.
-- Business value: Powers the top-line KPI cards on the Power BI dashboard.
SELECT
    -- App KPIs
    COUNT(*)                                     AS total_apps,
    COUNT(DISTINCT category)                     AS total_categories,
    COUNT(DISTINCT genres)                        AS total_genres,
    SUM(CASE WHEN is_paid = 1 THEN 1 ELSE 0 END) AS paid_apps,
    SUM(CASE WHEN is_paid = 0 THEN 1 ELSE 0 END) AS free_apps,

    -- Rating KPIs
    ROUND(AVG(rating), 2)                        AS avg_rating,
    ROUND(MIN(rating), 2)                        AS min_rating,
    ROUND(MAX(rating), 2)                        AS max_rating,

    -- Install KPIs
    SUM(installs_count)                          AS total_installs,
    ROUND(AVG(installs_count), 0)                AS avg_installs,
    MAX(installs_count)                          AS max_installs,

    -- Pricing KPIs
    ROUND(AVG(CASE WHEN is_paid = 1 THEN price_usd END), 2) AS avg_paid_price,
    ROUND(MAX(price_usd), 2)                     AS max_price,

    -- Size KPIs
    ROUND(AVG(size_in_mb), 2)                    AS avg_size_mb,
    ROUND(MAX(size_in_mb), 2)                    AS max_size_mb
FROM dim_apps;

-- Q36 | Sentiment KPI summary – single-row snapshot of the review corpus.
-- Business value: Powers the sentiment section of the executive dashboard.
SELECT COUNT(*)                                                      AS total_reviews,
    SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END)         AS positive_reviews,
    SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END)         AS negative_reviews,
    SUM(CASE WHEN sentiment = 'Neutral'  THEN 1 ELSE 0 END)         AS neutral_reviews,
    ROUND(
        SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(SUM(CASE WHEN sentiment IS NOT NULL THEN 1 ELSE 0 END), 0),
        2
    )                                                                AS positive_pct,
    ROUND(AVG(sentiment_polarity), 4)                                AS avg_polarity,
    ROUND(AVG(sentiment_subjectivity), 4)                            AS avg_subjectivity,
    COUNT(DISTINCT app_id)                                           AS apps_with_reviews
FROM fact_user_reviews;

-- Q37 | Category-level KPI matrix for the dashboard drill-down view.
-- Business value: Enables category comparison on a single dashboard page.
SELECT d.category,
    COUNT(DISTINCT d.app_id)                     AS total_apps,
    ROUND(AVG(d.rating), 2)                      AS avg_rating,
    SUM(d.installs_count)                        AS total_installs,
    COUNT(f.review_id)                           AS total_reviews,
    ROUND(
        SUM(CASE WHEN f.sentiment = 'Positive' THEN 1 ELSE 0 END)
        * 100.0
        / NULLIF(SUM(CASE WHEN f.sentiment IS NOT NULL THEN 1 ELSE 0 END), 0),
        2
    )                                            AS positive_review_pct,
    ROUND(AVG(f.sentiment_polarity), 4)          AS avg_polarity,
    ROUND(AVG(d.price_usd), 2)                   AS avg_price
FROM dim_apps d
    LEFT JOIN fact_user_reviews f ON d.app_id = f.app_id
GROUP BY d.category
ORDER BY total_installs DESC;

-- ============================================================================
-- END OF ANALYTICS QUERIES
-- ============================================================================