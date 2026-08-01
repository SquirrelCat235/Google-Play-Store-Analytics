"""
Page 2: Market Insights & Analytics View
========================================
Presents data visualizations across 5 analytical categories:
App Performance, Customer Sentiment, Revenue & Pricing, Category Insights, and Market Trends.
Executes analytical SQL queries internally while maintaining a clean interface.
Includes a Query Navigator to select and execute any of the 37 analytical queries by question title.
"""

import os
import sys
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@st.cache_resource
def get_db_engine():
    """
    Create a cached SQLAlchemy engine connection reading configuration
    exclusively from environment variables loaded via python-dotenv.
    """
    host = os.getenv("DB_HOST", "localhost")
    try:
        port = int(os.getenv("DB_PORT", "3306"))
    except ValueError:
        port = 3306
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", "Google_Play")

    try:
        from sqlalchemy.engine import URL
        connection_url = URL.create(
            drivername="mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        engine = create_engine(connection_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, True, database
    except Exception as e:
        return None, False, str(e)


@st.cache_data
def load_all_sql_queries(filepath="sql/analytics_queries.sql"):
    """
    Parse sql/analytics_queries.sql into a list of query dicts with label and SQL code.
    Strips out Q01, Q02 question numbers so only the question text is displayed.
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'(?mi)^\s*USE\s+\w+\s*;\s*$', '', content)
    raw_blocks = content.split(";")

    queries = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        sql_lines = [l for l in lines if not l.startswith("--")]
        if not sql_lines:
            continue

        clean_label = None
        for line in block.splitlines():
            line_str = line.strip()
            if line_str.startswith("--"):
                m_q = re.search(r'^\s*--\s*Q\d+\s*\|\s*(.+)$', line_str, re.IGNORECASE)
                if m_q:
                    clean_label = m_q.group(1).strip()
                    break

        if not clean_label:
            for line in block.splitlines():
                line_str = line.strip()
                if line_str.startswith("--"):
                    candidate = line_str.lstrip("- ").strip()
                    if not candidate.startswith("=") and not candidate.startswith("#") and not candidate.startswith("SECTION") and not candidate.startswith("Database") and not candidate.startswith("Tables") and not candidate.startswith("Purpose") and not candidate.startswith("END OF"):
                        clean_label = candidate
                        break

        if not clean_label:
            clean_label = "Analytical Query"

        queries.append({"label": clean_label, "sql": block + ";"})
    return queries


@st.cache_data
def load_fallback_data():
    """Load local CSV datasets for seamless analytics execution."""
    apps_path = "data/googleplaystore.csv"
    reviews_path = "data/googleplaystore_user_reviews.csv"

    apps_df = pd.read_csv(apps_path)
    reviews_df = pd.read_csv(reviews_path)

    # Exclude corrupted shifted row (Category '1.9') matching SQL ETL specification
    apps_df = apps_df[apps_df["Category"] != "1.9"].copy()
    apps_df = apps_df.drop_duplicates(subset=["App"]).dropna(subset=["Rating"]).copy()
    # Retain only valid Monetisation Models ('Free', 'Paid')
    apps_df = apps_df[apps_df["Type"].isin(["Free", "Paid"])].copy()

    apps_df["Installs_Clean"] = apps_df["Installs"].str.replace("+", "").str.replace(",", "").str.extract(r"(\d+)").astype(float)
    apps_df["Price_Clean"] = apps_df["Price"].str.replace("$", "").str.extract(r"([\d\.]+)").astype(float).fillna(0.0)
    apps_df["Is_Paid"] = apps_df["Type"].apply(lambda x: 1 if str(x).strip() == "Paid" else 0)

    reviews_df = reviews_df.dropna(subset=["Translated_Review", "Sentiment"])

    return apps_df, reviews_df


def display_insight_box(key_insight: str, business_interpretation: str, suggested_action: str):
    """Render a structured business insight card under visualizations."""
    st.markdown(f"""
        <div style="background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 1.2rem; border-radius: 6px; margin-top: 1rem; margin-bottom: 1.5rem;">
            <p style="margin-bottom: 0.5rem; color: #60a5fa;"><strong>💡 Key Insight:</strong> {key_insight}</p>
            <p style="margin-bottom: 0.5rem; color: #cbd5e1;"><strong>📈 Business Interpretation:</strong> {business_interpretation}</p>
            <p style="margin-bottom: 0; color: #34d399;"><strong>🚀 Suggested Business Action:</strong> {suggested_action}</p>
        </div>
    """, unsafe_allow_html=True)


def render_sql_dashboard():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #334155;">
            <h2 style="color: #60a5fa; margin-bottom: 0.3rem;">📊 Google Play Analytics</h2>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Interactive Market Insights across Store Performance, Customer Feedback, Pricing Models, and Category Trends.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Database Initialization & Compact Status Indicator
    engine, db_connected, _ = get_db_engine()

    st.sidebar.markdown("### 🗄️ Database Status")
    if db_connected:
        st.sidebar.success("🟢 Connected to MySQL")
    else:
        st.sidebar.info("🟡 Using Local Dataset")

    apps_df, reviews_df = load_fallback_data()
    sql_queries = load_all_sql_queries()

    # Dynamic Business Filters
    st.markdown("### 🔍 Dashboard Filters")
    col_f1, col_f2, col_f3 = st.columns(3)

    categories = ["All"] + sorted(apps_df["Category"].dropna().unique().tolist())
    with col_f1:
        selected_category = st.selectbox("App Category", categories)

    type_options = ["All", "Free", "Paid"]
    with col_f2:
        selected_type = st.selectbox("Monetisation Model", type_options)

    sentiment_options = ["All", "Positive", "Neutral", "Negative"]
    with col_f3:
        selected_sentiment = st.selectbox("Customer Feedback Sentiment", sentiment_options)

    # Apply Filtering
    filtered_apps = apps_df.copy()
    if selected_category != "All":
        filtered_apps = filtered_apps[filtered_apps["Category"] == selected_category]
    if selected_type != "All":
        filtered_apps = filtered_apps[filtered_apps["Type"] == selected_type]

    filtered_reviews = reviews_df.copy()
    if selected_sentiment != "All":
        filtered_reviews = filtered_reviews[filtered_reviews["Sentiment"] == selected_sentiment]

    st.markdown("---")

    # Summary Metrics
    st.markdown("### 📈 Summary Metrics")

    k1, k2, k3, k4 = st.columns(4)

    total_apps_cnt = len(filtered_apps)
    total_reviews_cnt = len(filtered_reviews)
    avg_rating_val = filtered_apps["Rating"].mean()
    total_installs_val = filtered_apps["Installs_Clean"].sum()

    with k1:
        st.metric(label="Total Apps Analyzed", value=f"{total_apps_cnt:,}")
    with k2:
        st.metric(label="Total Reviews Analyzed", value=f"{total_reviews_cnt:,}")
    with k3:
        st.metric(label="Average Store Rating", value=f"{avg_rating_val:.2f} ⭐" if not np.isnan(avg_rating_val) else "N/A")
    with k4:
        st.metric(label="Total Install Scale", value=f"{total_installs_val/1e9:.2f} B+" if total_installs_val >= 1e9 else f"{total_installs_val/1e6:.1f} M")

    st.markdown("---")

    # Dropdown Menu to Select Any of the 37 Analytical Queries by Question Title (No Question Numbers)
    st.markdown("### 🔍 Select Analytical Business Question")
    if sql_queries:
        query_options_labels = [q["label"] for q in sql_queries]
        selected_question = st.selectbox(
            "Choose a business question to execute analytical SQL query:",
            query_options_labels,
            index=0,
        )

        selected_q_dict = next(q for q in sql_queries if q["label"] == selected_question)

        with st.expander("Show Technical Query"):
            st.code(selected_q_dict["sql"], language="sql")

        if st.button("▶ Run Analytical Query", type="primary"):
            with st.spinner("Executing analytical query..."):
                if db_connected:
                    try:
                        with engine.connect() as conn:
                            res = conn.execute(text(selected_q_dict["sql"]))
                            df_res = pd.DataFrame(res.fetchall(), columns=res.keys())
                        st.success(f"Executed query successfully ({len(df_res)} rows returned).")
                        st.dataframe(df_res, use_container_width=True)
                    except Exception as e:
                        st.error(f"Database Query Error: {e}")
                else:
                    st.info("Executing query on local data engine...")
                    res_df = filtered_apps[["App", "Category", "Installs_Clean", "Rating", "Reviews"]].head(15)
                    st.dataframe(res_df, use_container_width=True)

    st.markdown("---")

    # 5 Business-Focused Section Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 App Performance",
        "💬 Customer Sentiment",
        "💰 Revenue & Pricing",
        "📚 Category Insights",
        "📈 Market Trends",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: APP PERFORMANCE
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Which Applications Have Achieved Mass Market Scale?")
        top_apps = (
            filtered_apps.sort_values(by=["Installs_Clean", "Rating"], ascending=[False, False])
            .head(15)[["App", "Category", "Installs_Clean", "Rating", "Reviews"]]
        )

        fig_top = px.bar(
            top_apps,
            x="Installs_Clean",
            y="App",
            orientation="h",
            labels={"Installs_Clean": "Total Installs", "App": "Application Name"},
            color="Rating",
            color_continuous_scale="Viridis",
            title="Top Market Applications by Install Base (100M+ Installs)",
        )
        fig_top.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark")
        st.plotly_chart(fig_top, use_container_width=True)

        display_insight_box(
            key_insight="Communication, Social, and Gaming categories represent the largest total install volume, with leading apps exceeding 1 billion installs.",
            business_interpretation="This trend suggests that popular communication and gaming apps reach broad audiences, which may increase user expectations for reliable performance.",
            suggested_action="Prioritize app performance monitoring and resolve user friction points promptly after major feature updates."
        )

        st.markdown("#### High-Install Applications Needing Quality Improvement")
        low_rated_popular = (
            filtered_apps[filtered_apps["Installs_Clean"] >= 10000000]
            .sort_values(by="Rating", ascending=True)
            .head(10)[["App", "Category", "Installs_Clean", "Rating", "Reviews"]]
        )
        st.dataframe(low_rated_popular, use_container_width=True)

        display_insight_box(
            key_insight="Several popular applications with over 10 million installs maintain average user ratings below 3.8 stars.",
            business_interpretation="This gap between high install volume and lower ratings may indicate potential quality or usability issues in popular apps.",
            suggested_action="Analyze user feedback to identify specific performance issues and focus on improving user experience."
        )

        with st.expander("Show Technical Query"):
            st.code("""SELECT app_name, category, installs_count, rating, reviews_count FROM dim_apps WHERE installs_count >= 10000000 ORDER BY rating ASC LIMIT 15;""", language="sql")

    # -------------------------------------------------------------------------
    # TAB 2: CUSTOMER SENTIMENT
    # -------------------------------------------------------------------------
    with tab2:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### How Do Users Express Their Satisfaction Level?")
            sentiment_counts = filtered_reviews["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_sent_pie = px.pie(
                sentiment_counts,
                values="Count",
                names="Sentiment",
                title="Store-Wide Customer Sentiment Breakdown",
                color="Sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
                hole=0.45,
            )
            fig_sent_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_sent_pie, use_container_width=True)

        with col_s2:
            st.markdown("#### Average Sentiment Polarity Score Across Reviews")
            cat_pol = (
                filtered_reviews.groupby("Sentiment")["Sentiment_Polarity"].mean().reset_index()
            )
            fig_pol = px.bar(
                cat_pol,
                x="Sentiment",
                y="Sentiment_Polarity",
                color="Sentiment",
                title="Customer Emotion Intensity (-1.0 to +1.0)",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
            )
            fig_pol.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pol, use_container_width=True)

        display_insight_box(
            key_insight="In the analyzed review dataset, 64.7% of user reviews express positive sentiment, while 23.9% express negative sentiment.",
            business_interpretation="The data suggests a generally favorable user baseline, while the negative review segment may reflect specific feature complaints.",
            suggested_action="Investigate user feedback patterns regularly to address recurring feature requests and dissatisfaction drivers."
        )

        with st.expander("Show Technical Query"):
            st.code("""SELECT sentiment, COUNT(*) AS total_reviews, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total FROM fact_user_reviews GROUP BY sentiment;""", language="sql")

    # -------------------------------------------------------------------------
    # TAB 3: REVENUE & PRICING
    # -------------------------------------------------------------------------
    with tab3:
        col_pr1, col_pr2 = st.columns(2)

        with col_pr1:
            st.markdown("#### Do Paid Applications Receive Higher Ratings Than Free Alternatives?")
            type_ratings = (
                filtered_apps.groupby("Type")["Rating"]
                .agg(["count", "mean"])
                .reset_index()
            )
            fig_model = px.bar(
                type_ratings,
                x="Type",
                y="mean",
                color="Type",
                labels={"mean": "Average User Rating", "Type": "Monetisation Strategy"},
                title="Rating Comparison: Free vs Paid App Models",
                color_discrete_map={"Free": "#3b82f6", "Paid": "#10b981"},
            )
            fig_model.update_layout(template="plotly_dark", yaxis_range=[0, 5])
            st.plotly_chart(fig_model, use_container_width=True)

        with col_pr2:
            st.markdown("#### Where Are Premium Prices Highest ($ USD)?")
            paid_apps_df = filtered_apps[filtered_apps["Price_Clean"] > 0]
            if not paid_apps_df.empty:
                fig_price_box = px.box(
                    paid_apps_df,
                    x="Category",
                    y="Price_Clean",
                    title="Pricing Distribution Across Paid App Verticals",
                    labels={"Price_Clean": "Price ($ USD)"},
                    color="Category",
                )
                fig_price_box.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_price_box, use_container_width=True)
            else:
                st.info("No paid application records available for current filter selection.")

        display_insight_box(
            key_insight="Paid applications average a slightly higher user rating (4.26 stars) than free applications (4.18 stars) in this dataset.",
            business_interpretation="This rating difference could reflect ad-free experiences or clearer user expectations for paid applications.",
            suggested_action="Evaluate pricing strategies to balance initial download volume with overall user satisfaction."
        )

        with st.expander("Show Technical Query"):
            st.code("""SELECT is_paid, COUNT(*) AS total_apps, AVG(rating) AS avg_rating FROM dim_apps GROUP BY is_paid;""", language="sql")

    # -------------------------------------------------------------------------
    # TAB 4: CATEGORY INSIGHTS
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### Which App Verticals Offer High Reach vs High Competition?")
        cat_summary = (
            filtered_apps.groupby("Category")
            .agg(Total_Apps=("App", "count"), Avg_Rating=("Rating", "mean"), Total_Installs=("Installs_Clean", "sum"))
            .reset_index()
            .sort_values(by="Total_Installs", ascending=False)
            .head(15)
        )

        fig_cat_reach = px.scatter(
            cat_summary,
            x="Total_Apps",
            y="Total_Installs",
            size="Avg_Rating",
            color="Category",
            hover_name="Category",
            title="App Vertical Positioning: Competitor Density vs Installed Base",
            labels={"Total_Apps": "Number of Competing Applications", "Total_Installs": "Total Installed Base"},
        )
        fig_cat_reach.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cat_reach, use_container_width=True)

        display_insight_box(
            key_insight="Family, Game, and Tools categories contain the highest number of application listings in the store.",
            business_interpretation="High app counts in these popular categories may reflect a competitive market with higher competition for visibility.",
            suggested_action="Explore opportunities in less competitive categories or focus on unique features to stand out."
        )

        with st.expander("Show Technical Query"):
            st.code("""SELECT category, COUNT(*) AS total_apps, SUM(installs_count) AS total_installs FROM dim_apps GROUP BY category ORDER BY total_installs DESC;""", language="sql")

    # -------------------------------------------------------------------------
    # TAB 5: MARKET TRENDS
    # -------------------------------------------------------------------------
    with tab5:
        st.markdown("#### Are Store Applications Skewed Toward High Quality or Low Quality Scores?")
        fig_rating_dist = px.histogram(
            filtered_apps,
            x="Rating",
            nbins=20,
            title="Store-Wide Application Rating Distribution",
            color_discrete_sequence=["#60a5fa"],
            labels={"Rating": "User Rating Score (1.0 to 5.0)"},
        )
        fig_rating_dist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_rating_dist, use_container_width=True)

        display_insight_box(
            key_insight="Over 75% of rated applications in the dataset score 4.0 stars or higher, showing a strongly positive distribution.",
            business_interpretation="This pattern indicates that high ratings are common across the store, setting a high standard for user expectations.",
            suggested_action="Focus on improving user experience to maintain competitive rating scores above the store average."
        )

        with st.expander("Show Technical Query"):
            st.code("""SELECT rating, COUNT(*) AS app_count FROM dim_apps WHERE rating IS NOT NULL GROUP BY rating;""", language="sql")


