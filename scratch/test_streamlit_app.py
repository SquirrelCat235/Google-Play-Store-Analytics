"""
Headless verification script for refactored Streamlit app pages.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from views.home import render_home_page
from views.sql_dashboard import get_db_engine, load_fallback_data
from views.single_prediction import load_distilbert_pipeline
from views.about import render_about_page

def run_tests():
    print("==================================================")
    print(" Refactored Streamlit App Headless Verification  ")
    print("==================================================")

    # 1. Test fallback data loader
    print("\n1. Testing Business Insights Data Engine Loader...")
    apps_df, reviews_df = load_fallback_data()
    print(f"   - Apps dataset shape   : {apps_df.shape}")
    print(f"   - Reviews dataset shape: {reviews_df.shape}")
    assert not apps_df.empty, "Apps DataFrame is empty!"
    assert not reviews_df.empty, "Reviews DataFrame is empty!"
    print("   [OK] Data Engine Loader PASSED!")

    # 2. Test DistilBERT pipeline loading
    print("\n2. Testing AI Sentiment Pipeline Loader...")
    tokenizer, model, err = load_distilbert_pipeline()
    assert err is None, f"DistilBERT loading failed: {err}"
    assert tokenizer is not None, "Tokenizer is None!"
    assert model is not None, "Model is None!"
    print("   [OK] AI Sentiment Pipeline Loader PASSED!")

    # 3. Test DistilBERT real-time inference
    print("\n3. Testing Real-Time Forward Sentiment Inference...")
    import torch
    import torch.nn.functional as F
    from src.utils import get_device
    from src.preprocessing import ID2LABEL

    device = get_device()
    test_review = "This Google Play Store app is fast, intuitive, and extremely helpful!"
    inputs = tokenizer(test_review, truncation=True, padding="max_length", max_length=64, return_tensors="pt").to(device)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
        pred_id = int(torch.argmax(outputs.logits, dim=1).item())

    pred_sentiment = ID2LABEL[pred_id]
    print(f"   - Input Text      : '{test_review}'")
    print(f"   - Predicted Class : {pred_sentiment} (Confidence: {probs[pred_id]*100:.2f}%)")
    assert pred_sentiment == "Positive", f"Expected Positive sentiment, got {pred_sentiment}"
    print("   [OK] Sentiment Inference PASSED!")

    print("\n==================================================")
    print(" [SUCCESS] ALL REFACTORED APP VERIFICATION TESTS PASSED ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
