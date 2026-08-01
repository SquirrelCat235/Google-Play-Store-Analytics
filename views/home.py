"""
Page 1: Home Page
=================
Presents a project overview, summary metrics, key analytical topics, and navigation cards.
"""

import streamlit as st

def render_home_page():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #334155;">
            <h1 style="color: #60a5fa; margin-bottom: 0.5rem; font-family: 'Inter', sans-serif;">📱 Google Play Analytics</h1>
            <p style="color: #94a3b8; font-size: 1.15rem; margin-bottom: 0;">
                Exploratory Data Analysis and NLP Sentiment Classification on Google Play Store Data.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 4 KPI Cards
    st.markdown("### 📊 Dataset & Model Performance Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Apps Analyzed", value="10,841", delta="Apps Dataset")
    with col2:
        st.metric(label="Total User Reviews", value="27,992", delta="Reviews Dataset")
    with col3:
        st.metric(label="Average App Rating", value="4.19 ⭐", delta="Store Benchmark")
    with col4:
        st.metric(label="DistilBERT Test Macro F1", value="0.9066", delta="Accuracy: 92.12%")

    st.markdown("---")

    # What can you learn section
    st.markdown("### 💡 What Does This Project Cover?")
    c_learn1, c_learn2 = st.columns(2)

    with c_learn1:
        st.markdown("""
        - 📈 **Market Leaders & Category Distribution**: Analyze which app categories drive total install volumes and examine competition levels.
        - 💰 **Pricing Models**: Compare user rating metrics between free and paid applications.
        """)

    with c_learn2:
        st.markdown("""
        - 💬 **Customer Sentiment Analysis**: Examine sentiment distributions and polarity scores across user review text.
        - 🤖 **Automated Sentiment Classification**: Use a fine-tuned DistilBERT model to classify review text into sentiment categories.
        """)

    st.markdown("---")

    # Navigation Cards
    st.markdown("### 🧭 Project Modules")
    st.markdown("Select a module below or use the sidebar navigation to explore:")

    n1, n2, n3, n4 = st.columns(4)

    with n1:
        st.info("📊 **Business Insights**\n\nExplore app performance, customer sentiment, category dynamics, pricing, and market trends.")
    with n2:
        st.info("🤖 **Predict Review**\n\nAnalyze customer sentiment in real time for individual Google Play app reviews.")
    with n3:
        st.info("📁 **Batch Analysis**\n\nUpload review datasets to generate automated sentiment reports and download predictions.")
    with n4:
        st.info("ℹ️ **About Project**\n\nReview project overview and technical implementation details.")

