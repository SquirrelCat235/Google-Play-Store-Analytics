"""
Google Play Store Analytics – Data Analytics & Sentiment Classification Project
================================================================================
Multi-page Streamlit web application combining market analytics and NLP review sentiment prediction.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Configure Streamlit Page Settings
st.set_page_config(
    page_title="Google Play Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import View Modules
from views.home import render_home_page
from views.sql_dashboard import render_sql_dashboard
from views.single_prediction import render_single_prediction_page
from views.batch_prediction import render_batch_prediction_page
from views.about import render_about_page


# Inject Dark Theme CSS
st.markdown("""
    <style>
        /* Main background and font styling */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        /* Sidebar header spacing */
        div[data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        
        /* Metric Card styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            padding: 1.2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #60a5fa !important;
            font-weight: 700 !important;
        }

        /* Buttons styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-weight: 600;
            color: #94a3b8;
        }
        
        button[aria-selected="true"] {
            color: #60a5fa !important;
            border-bottom-color: #3b82f6 !important;
        }
    </style>
""", unsafe_allow_html=True)


def main():
    # Sidebar Header & Brand
    st.sidebar.markdown("""
        <div style="text-align: center; padding-bottom: 1rem; border-bottom: 1px solid #1e293b;">
            <h2 style="color: #60a5fa; margin-bottom: 0;">📱 Google Play Analytics</h2>
            <p style="color: #64748b; font-size: 0.85rem;">Data Analytics & NLP Project</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation Menu
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        label="Select Page",
        options=[
            "🏠 Home",
            "📊 Business Insights",
            "🤖 Predict Review",
            "📁 Batch Analysis",
            "ℹ️ About Project",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Google Play Store Analytics Project")


    # Route Selected Page
    if "🏠 Home" in page:
        render_home_page()
    elif "📊 Business Insights" in page:
        render_sql_dashboard()
    elif "🤖 Predict Review" in page:
        render_single_prediction_page()
    elif "📁 Batch Analysis" in page:
        render_batch_prediction_page()
    elif "ℹ️ About Project" in page:
        render_about_page()


if __name__ == "__main__":
    main()
