"""
Page 4: Batch Analysis View
===========================
Upload a CSV file containing user review texts, execute AI batch sentiment classification,
display executive summary metrics (% Positive, % Neutral, % Negative), render an interactive dataframe,
and generate a short business summary of the uploaded dataset alongside downloadable CSV outputs.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.express as px
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import get_device
from src.preprocessing import ID2LABEL
from views.single_prediction import load_distilbert_pipeline
from src.classifier import ReviewEvalDataset


def render_batch_prediction_page():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #334155;">
            <h2 style="color: #60a5fa; margin-bottom: 0.3rem;">📁 Batch Customer Review Analysis</h2>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Upload customer review datasets to analyze customer sentiment at scale and download prediction reports.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    tokenizer, model, err = load_distilbert_pipeline()
    if err:
        st.error(f"Failed to load Sentiment Model: {err}")
        return

    st.markdown("### 📥 Upload Customer Reviews Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file (Must contain a column with review texts)", type=["csv"])

    # Provide Sample CSV Download Button
    st.markdown("**Need a sample dataset?**")
    if os.path.exists("outputs/cleaned_data/test.csv"):
        with open("outputs/cleaned_data/test.csv", "rb") as f:
            st.download_button(
                label="📥 Download Sample Customer Reviews CSV (4,199 rows)",
                data=f,
                file_name="sample_customer_reviews.csv",
                mime="text/csv",
            )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")
            st.dataframe(df.head(5), use_container_width=True)

            # Select column containing review text
            possible_cols = [c for c in df.columns if any(k in c.lower() for k in ["review", "text", "comment"])]
            default_col = possible_cols[0] if possible_cols else df.columns[0]
            text_column = st.selectbox("Select Column Containing Customer Review Text", df.columns, index=df.columns.get_loc(default_col))

            if st.button("🚀 Execute Batch Analysis", type="primary", use_container_width=True):
                texts = df[text_column].astype(str).tolist()

                st.info(f"Processing {len(texts):,} reviews using DistilBERT model...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                device = get_device()
                eval_dataset = ReviewEvalDataset(texts, [0] * len(texts), tokenizer, max_length=64)
                eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

                all_preds = []
                all_confidences = []
                start_time = time.time()

                with torch.no_grad():
                    for batch_idx, batch in enumerate(eval_loader):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        if torch.cuda.is_available():
                            with torch.amp.autocast("cuda"):
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        else:
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                        probs = F.softmax(outputs.logits, dim=1)
                        conf, preds = torch.max(probs, dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_confidences.extend(conf.cpu().numpy())

                        progress = (batch_idx + 1) / len(eval_loader)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed batch {batch_idx + 1} of {len(eval_loader)} ({int(progress * 100)}%)...")

                total_time = time.time() - start_time
                status_text.empty()
                progress_bar.empty()

                # Build Output DataFrame
                df_out = df.copy()
                df_out["predicted_sentiment"] = [ID2LABEL[p] for p in all_preds]
                df_out["confidence_score"] = [float(f"{c*100:.2f}") for c in all_confidences]

                # Compute Metrics
                total_cnt = len(df_out)
                pos_cnt = (df_out["predicted_sentiment"] == "Positive").sum()
                neu_cnt = (df_out["predicted_sentiment"] == "Neutral").sum()
                neg_cnt = (df_out["predicted_sentiment"] == "Negative").sum()

                pos_pct = (pos_cnt / total_cnt) * 100.0
                neu_pct = (neu_cnt / total_cnt) * 100.0
                neg_pct = (neg_cnt / total_cnt) * 100.0

                st.markdown("---")
                st.markdown("### 📊 Dataset Sentiment Summary")

                # Required Summary Metrics Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(label="Total Reviews Processed", value=f"{total_cnt:,}")
                with m2:
                    st.metric(label="Positive Sentiment %", value=f"{pos_pct:.1f}%")
                with m3:
                    st.metric(label="Neutral Sentiment %", value=f"{neu_pct:.1f}%")
                with m4:
                    st.metric(label="Negative Sentiment %", value=f"{neg_pct:.1f}%")

                st.markdown("---")

                # Short Summary of Uploaded Dataset
                st.markdown("### 📋 Sentiment Summary & Observations")

                if pos_pct >= 60.0:
                    health_status = "🟢 Predominantly Positive Feedback"
                    health_desc = f"Over {pos_pct:.1f}% of reviews express positive user sentiment."
                elif neg_pct >= 30.0:
                    health_status = "🔴 Higher Concentration of Negative Feedback"
                    health_desc = f"Noticeable concentration of negative reviews ({neg_pct:.1f}%)."
                else:
                    health_status = "🟡 Balanced Sentiment Distribution"
                    health_desc = f"Balanced sentiment mix ({pos_pct:.1f}% positive, {neg_pct:.1f}% negative)."

                st.info(f"**Dataset Sentiment Distribution**: {health_status}\n\n**Analysis**: {health_desc}")


                b_col1, b_col2 = st.columns([1.2, 0.8])

                with b_col1:
                    st.markdown("#### Annotated Prediction Dataset")
                    st.dataframe(df_out, use_container_width=True)

                with b_col2:
                    st.markdown("#### Sentiment Proportion")
                    sentiment_counts = df_out["predicted_sentiment"].value_counts().reset_index()
                    sentiment_counts.columns = ["Sentiment", "Count"]

                    fig_pie = px.pie(
                        sentiment_counts,
                        values="Count",
                        names="Sentiment",
                        title="Uploaded Dataset Sentiment Breakdown",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
                        hole=0.4,
                    )
                    fig_pie.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Required Download Button
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Annotated Predictions CSV Report",
                    data=csv_bytes,
                    file_name="batch_sentiment_predictions_report.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as e:
            st.error(f"Error processing uploaded CSV file: {e}")
