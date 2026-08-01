"""
Page 5: About Project View
==========================
Presents a non-technical Project Overview for business stakeholders,
alongside an expandable Technical Details section for technical reviewers and recruiters.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def render_about_page():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #334155;">
            <h2 style="color: #60a5fa; margin-bottom: 0.3rem;">ℹ️ About Google Play Analytics</h2>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Project Overview, Analytical Workflow, and Machine Learning Benchmarks.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # SECTION 1: PROJECT OVERVIEW (NON-TECHNICAL)
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    This project provides data-driven market insights and machine learning sentiment classification for Google Play Store app reviews.

    **Core Capabilities:**
    - 📊 **Market Insights**: Interactive dashboards tracking store-wide app performance, pricing models, rating distributions, and category trends.
    - 💬 **Customer Feedback Analysis**: Natural language sentiment analysis to evaluate user satisfaction across 37,000+ app reviews.
    - 🤖 **Automated Sentiment Classification**: Fine-tuned DistilBERT transformer model for real-time and batch review sentiment prediction.
    """)

    st.markdown("---")


    # SECTION 2: TECHNICAL DETAILS (EXPANDABLE)
    st.markdown("### 🛠️ Technical Details")
    st.caption("Expand the section below for engineering architecture, machine learning benchmarks, and code implementation specs.")

    with st.expander("🔬 Show Technical Details (For Recruiters & Technical Reviewers)", expanded=False):
        st.markdown("#### 1. System Architecture & ETL Workflow")
        st.markdown("""
        - **Relational Data Warehouse**: MySQL 8.0 Star Schema (`dim_apps` dimension table and `fact_user_reviews` fact table).
        - **Data Pipeline**: Cleaned, deduplicated, and normalized 10,841 store applications and 37,427 user reviews.
        - **Analytical SQL Suite**: 37 business intelligence queries covering dense rankings, composite scores, and category metrics.
        """)

        st.markdown("---")

        st.markdown("#### 2. Model Development Workflow")
        st.markdown("""
        - **Preprocessed Corpus**: 27,992 valid labeled reviews split into Train (70%, 19,594 samples), Validation (15%, 4,199 samples), and Test (15%, 4,199 samples).
        - **Baselines**: Logistic Regression and Linear SVM with 5-fold `GridSearchCV` hyperparameter tuning on TF-IDF n-gram features (20,000 max features).
        - **Deep Learning Transformer**: Fine-tuned `distilbert-base-uncased` on NVIDIA GeForce RTX 2050 GPU using PyTorch 2.6+CUDA 12.4 with Automatic Mixed Precision (AMP `BF16`/`FP16`).
        - **Class Imbalance Management**: Weighted `CrossEntropyLoss` ($w = [1.50, 2.25, 0.53]$).
        - **Optimizer & Scheduler**: `AdamW` (lr = 2e-5, weight_decay = 0.01) with 10% linear warmup and Early Stopping monitoring.
        """)

        st.markdown("---")

        st.markdown("#### 3. Multi-Model Performance Comparison Table")
        st.markdown("Evaluated on 4,199 held-out test split reviews:")

        benchmark_data = [
            {
                "Model": "Logistic Regression",
                "Accuracy": 0.8590,
                "Macro Precision": 0.8120,
                "Macro Recall": 0.8443,
                "Macro F1 (Primary)": 0.8266,
                "Weighted F1": 0.8610,
                "ROC-AUC (OvR)": 0.9520,
                "Training Time": "0m 16s",
                "Inference Time (s)": 0.0050,
            },
            {
                "Model": "Linear SVM",
                "Accuracy": 0.8609,
                "Macro Precision": 0.8186,
                "Macro Recall": 0.8340,
                "Macro F1 (Primary)": 0.8258,
                "Weighted F1": 0.8618,
                "ROC-AUC (OvR)": 0.9530,
                "Training Time": "0m 04s",
                "Inference Time (s)": 0.0021,
            },
            {
                "Model": "DistilBERT (Fine-Tuned)",
                "Accuracy": 0.9212,
                "Macro Precision": 0.9029,
                "Macro Recall": 0.9122,
                "Macro F1 (Primary)": 0.9066,
                "Weighted F1": 0.9221,
                "ROC-AUC (OvR)": 0.9819,
                "Training Time": "5m 19s",
                "Inference Time (s)": 12.5365,
            },
        ]

        df_comp = pd.DataFrame(benchmark_data)
        st.dataframe(
            df_comp.style.highlight_max(subset=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1 (Primary)", "Weighted F1", "ROC-AUC (OvR)"], color="#10b98130"),
            use_container_width=True,
        )

        st.markdown("---")

        st.markdown("#### 4. Model Selection Rationale")
        st.success("""
        **Selected Model: Fine-Tuned DistilBERT Transformer (`distilbert-base-uncased`)**

        - **Primary Metric Victory**: Achieves **`0.9066` Test Macro F1**, outperforming Logistic Regression (`0.8266`) and Linear SVM (`0.8258`) by **+8.0% absolute Macro F1 score**.
        - **Classification Accuracy**: Reaches **`92.12%`** overall test accuracy across 4,199 held-out reviews.
        - **Contextual Self-Attention**: Captures complex word order dependencies, context shifts, and subtle user sentiment negations (*"not bad at all"* vs *"bad update"*).
        """)

        st.markdown("---")

        st.markdown("#### 5. Saved Confusion Matrices")
        cm_col1, cm_col2, cm_col3 = st.columns(3)
        cm_lr_path = "outputs/figures/confusion_matrix_logistic_regression.png"
        cm_svm_path = "outputs/figures/confusion_matrix_linear_svm.png"
        cm_distil_path = "outputs/figures/confusion_matrix_distilbert.png"

        with cm_col1:
            st.markdown("**Logistic Regression**")
            if os.path.exists(cm_lr_path):
                st.image(cm_lr_path, use_container_width=True)
        with cm_col2:
            st.markdown("**Linear SVM**")
            if os.path.exists(cm_svm_path):
                st.image(cm_svm_path, use_container_width=True)
        with cm_col3:
            st.markdown("**DistilBERT (Selected)**")
            if os.path.exists(cm_distil_path):
                st.image(cm_distil_path, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 6. Classification Reports")
        st.markdown("""
        ```text
        DistilBERT Classification Report (Test Set):
                      precision    recall  f1-score   support

            Negative     0.8213    0.9135    0.8649       936
             Neutral     0.9268    0.8926    0.9094       624
            Positive     0.9605    0.9307    0.9453      2639

            accuracy                         0.9212      4199
           macro avg     0.9029    0.9122    0.9066      4199
        weighted avg     0.9245    0.9212    0.9221      4199
        ```
        """)

        st.markdown("---")

        st.markdown("#### 7. Project Limitations & Future Enhancements")
        st.markdown("""
        - **Limitations**:
          - Max sequence length set to 64 tokens (covers 97.14% of reviews, but truncates very long paragraphs).
          - Dataset is limited to English language user reviews.
        - **Future Improvements**:
          - Fine-tune multi-lingual transformer (XLM-RoBERTa) for global store reviews.
          - Integrate Llama-3 / Mistral LLM via Ollama for multi-sentence review summary generation.
          - Package application into a Docker container for AWS ECS auto-scaling.
        """)
