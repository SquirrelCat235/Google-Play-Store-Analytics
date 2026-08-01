"""
Page 3: Predict Review View
============================
Loads saved DistilBERT model & tokenizer from outputs/models/distilbert_sentiment/
and performs real-time sentiment prediction for custom Google Play Store app reviews,
providing actionable business recommendations for product developers.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import get_device
from src.preprocessing import ID2LABEL


@st.cache_resource
def load_distilbert_pipeline(hf_repo_id="Samayita-23/google-play-distilbert-sentiment"):
    """
    Cache and load saved DistilBERT model and tokenizer.
    Prioritizes local model, falls back to Hugging Face Hub if missing (e.g. in deployment).
    """
    # Resolve local model path relative to the project root (not CWD)
    local_model_dir = os.path.join(str(project_root), "outputs", "models", "distilbert_sentiment")

    # Check if actual model weight files exist locally
    has_safetensors = os.path.isfile(os.path.join(local_model_dir, "model.safetensors"))
    has_bin = os.path.isfile(os.path.join(local_model_dir, "pytorch_model.bin"))
    use_local = has_safetensors or has_bin

    model_path = local_model_dir if use_local else hf_repo_id
    source_label = "Local Model" if use_local else "Hugging Face Hub"
    print(f"[Model Loader] Source: {source_label} | Path: {model_path}", flush=True)

    try:
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, f"Failed to load model from '{model_path}': {str(e)}"


def render_single_prediction_page():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #334155;">
            <h2 style="color: #60a5fa; margin-bottom: 0.3rem;">🤖 Predict Review Sentiment</h2>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Analyze customer emotion and feedback intent behind any Google Play Store review text.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    with st.spinner("Loading DistilBERT model..."):
        tokenizer, model, err = load_distilbert_pipeline()

    if err:
        st.error(f"Failed to load Sentiment Model: {err}")
        return

    st.markdown("### ✍️ Enter a Google Play Review")

    # Sample Quick Review Preset Buttons
    st.markdown("**Test Preset Examples:**")
    col_e1, col_e2, col_e3 = st.columns(3)
    sample_text = ""

    if col_e1.button("🌟 Positive Review"):
        sample_text = "Absolutely fantastic app! The user interface is so smooth and customer support solved my issue in minutes."
    if col_e2.button("😐 Neutral Review"):
        sample_text = "The app works fine after the latest update, but the new layout takes some time to get used to."
    if col_e3.button("🔴 Negative Review"):
        sample_text = "Worst update ever! It keeps crashing every 5 minutes and drains my battery completely. Fix this bug immediately!"

    # Text Area Input
    user_review = st.text_area(
        label="Review Text Input",
        value=sample_text,
        height=130,
        placeholder="Type or paste any Google Play Store app review here...",
    )

    if st.button("🔮 Predict Sentiment", type="primary", use_container_width=True):
        if not user_review.strip():
            st.warning("Please enter a Google Play review text before predicting.")
            return

        with st.spinner("Analyzing review text using DistilBERT..."):
            device = get_device()
            start_time = time.time()

            inputs = tokenizer(
                user_review,
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda"):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
                pred_id = int(torch.argmax(outputs.logits, dim=1).item())

            latency_ms = (time.time() - start_time) * 1000
            pred_sentiment = ID2LABEL[pred_id]
            confidence_pct = probs[pred_id] * 100.0

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results & Business Impact")

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            badge_color = "#10b981" if pred_sentiment == "Positive" else "#64748b" if pred_sentiment == "Neutral" else "#ef4444"
            st.markdown(f"""
                <div style="background-color: {badge_color}15; border: 2px solid {badge_color}; padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <h4 style="color: #94a3b8; margin-bottom: 0.3rem;">Predicted Customer Sentiment</h4>
                    <h1 style="color: {badge_color}; font-size: 2.4rem; margin-bottom: 0.5rem;">{pred_sentiment.upper()}</h1>
                    <p style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 0;">Model Confidence Score: <strong>{confidence_pct:.2f}%</strong></p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">Processing Latency: {latency_ms:.2f} ms</p>
                </div>
            """, unsafe_allow_html=True)


        with res_col2:
            st.markdown("#### Sentiment Probability Distribution")
            prob_df = pd.DataFrame({
                "Sentiment": [ID2LABEL[i] for i in range(3)],
                "Probability (%)": probs * 100.0,
            })

            fig_prob = px.bar(
                prob_df,
                x="Probability (%)",
                y="Sentiment",
                orientation="h",
                color="Sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
                text_auto=".2f",
            )
            fig_prob.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), xaxis_range=[0, 100])
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")

        # Explanation for App Developers & Product Teams
        st.markdown("### 💡 What Does This Prediction Mean for App Product Teams?")

        if pred_sentiment == "Positive":
            st.success("""
            **Positive Customer Intent Actionable Guide:**
            - **Product Value**: The user loves feature capability, performance, or customer experience.
            - **Recommended Action**: Trigger an in-app review rating prompt to convert this satisfied user into a 5-star public rating on Google Play.
            """)
        elif pred_sentiment == "Neutral":
            st.info("""
            **Neutral Customer Intent Actionable Guide:**
            - **Product Value**: The user offers feature feedback, UI suggestions, or routine inquiries without strong emotion.
            - **Recommended Action**: Route to product feedback queue for user experience refinement and backlog prioritization.
            """)
        else:
            st.error("""
            **Negative Customer Intent Actionable Guide:**
            - **Product Value**: The user is experiencing application crashes, payment issues, or broken features.
            - **Recommended Action**: High priority! Route directly to QA & technical support teams to resolve before the issue impacts store rank.
            """)
