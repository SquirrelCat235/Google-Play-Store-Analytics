"""
Classifier Evaluation Module for Google Play Store Sentiment Models.
====================================================================
Implements test set evaluation pipeline for classical machine learning models:
  - Metric computation: Macro F1 (primary), Accuracy, Precision, Recall, ROC-AUC (OvR), Confusion Matrix, Inference Time.
  - Prediction persistence: Exports predictions CSVs.
  - Evaluation artifact persistence: Confusion matrix plots and classification reports.
  - Model comparison & selection based on Macro F1 score.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Add project root to sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import load_model, format_time
from src.preprocessing import LABEL2ID, ID2LABEL


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def evaluate_sklearn_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a trained scikit-learn model on test data and compute performance metrics.

    Args:
        model: Trained scikit-learn model estimator.
        X_test: Test feature matrix (TF-IDF).
        y_test: True ground-truth test labels.
        model_name (str): Human-readable name of the model.

    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics, predictions, and report string.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ({len(y_test):,} samples) ---")

    # Time inference
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # Predict probabilities or calibrated decision scores for ROC-AUC
    y_scores = None
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        y_scores = softmax(raw_scores)

    # Compute metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))  # Primary Metric
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted"))

    # ROC-AUC (One-vs-Rest)
    if y_scores is not None:
        try:
            roc_auc_macro = float(roc_auc_score(y_test, y_scores, multi_class="ovr", average="macro"))
            roc_auc_weighted = float(roc_auc_score(y_test, y_scores, multi_class="ovr", average="weighted"))
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC for {model_name}: {e}")
            roc_auc_macro = np.nan
            roc_auc_weighted = np.nan
    else:
        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    cls_report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cls_report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "inference_time": inference_time,
        "confusion_matrix": cm,
        "classification_report_str": cls_report_str,
        "classification_report_dict": cls_report_dict,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }

    # Print summary metrics
    print(f"Results for {model_name}:")
    print(f"  Macro F1 (Primary)   : {macro_f1:.4f}")
    print(f"  Accuracy             : {acc:.4f}")
    print(f"  Weighted Precision   : {weighted_prec:.4f}")
    print(f"  Weighted Recall      : {weighted_rec:.4f}")
    print(f"  ROC-AUC (OvR Weighted): {roc_auc_weighted:.4f}")
    print(f"  Inference Time       : {inference_time:.4f}s ({inference_time*1000/len(y_test):.3f} ms/sample)")
    print("\nClassification Report:\n" + cls_report_str)

    return metrics


def save_predictions(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "outputs/predictions",
) -> str:
    """
    Save model predictions alongside ground-truth labels and review text to CSV.

    Args:
        df_test (pd.DataFrame): Test dataset DataFrame.
        y_pred (np.ndarray): Predicted numerical labels.
        model_name (str): Model name for filename demarcation.
        output_dir (str): Destination directory.

    Returns:
        str: Absolute path to saved predictions CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_out = df_test.copy()
    df_out["predicted_label"] = y_pred
    df_out["predicted_sentiment"] = [ID2LABEL.get(lbl, "Unknown") for lbl in y_pred]
    df_out["correct"] = df_out["label"] == df_out["predicted_label"]

    model_key = model_name.lower().replace(" ", "_")
    output_path = os.path.join(output_dir, f"{model_key}_test_predictions.csv")
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions for '{model_name}' -> '{output_path}' ({len(df_out):,} rows).")
    return output_path


def save_evaluation_artifacts(
    metrics: Dict[str, Any],
    pred_dir: str = "outputs/predictions",
    fig_dir: str = "outputs/figures",
) -> Dict[str, str]:
    """
    Save confusion matrix visualization PNG and classification report text/json files.

    Args:
        metrics (Dict[str, Any]): Dictionary returned by evaluate_sklearn_model().
        pred_dir (str): Directory for report text/csv output.
        fig_dir (str): Directory for figure PNG output.

    Returns:
        Dict[str, str]: Paths to saved evaluation artifacts.
    """
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    model_name = metrics["model_name"]
    model_key = model_name.lower().replace(" ", "_")
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]

    # 1. Save Confusion Matrix Plot (PNG)
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        cbar=True,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Sentiment Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Sentiment Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_fig_path = os.path.join(fig_dir, f"confusion_matrix_{model_key}.png")
    fig.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Save Classification Report (TXT & JSON)
    report_txt_path = os.path.join(pred_dir, f"{model_key}_classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(metrics["classification_report_str"])

    report_json_path = os.path.join(pred_dir, f"{model_key}_classification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics["classification_report_dict"], f, indent=4)

    # 3. Save Confusion Matrix CSV
    cm_csv_path = os.path.join(pred_dir, f"{model_key}_confusion_matrix.csv")
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(cm_csv_path)

    print(f"Saved evaluation artifacts for '{model_name}':")
    print(f"  - Confusion Matrix Plot : {cm_fig_path}")
    print(f"  - Classification Report  : {report_txt_path}")
    print(f"  - Report JSON Data       : {report_json_path}")
    print(f"  - Confusion Matrix CSV   : {cm_csv_path}")

    return {
        "cm_plot": cm_fig_path,
        "report_txt": report_txt_path,
        "report_json": report_json_path,
        "cm_csv": cm_csv_path,
    }


def compare_models(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Construct side-by-side comparison DataFrame table of all evaluated models,
    display formatted table, and print best-performing model justification based on Macro F1.

    Args:
        metrics_list (List[Dict[str, Any]]): List of metric dictionaries from evaluate_sklearn_model().

    Returns:
        pd.DataFrame: Summary metrics DataFrame.
    """
    records = []
    for m in metrics_list:
        records.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Weighted Precision": m["weighted_precision"],
            "Weighted Recall": m["weighted_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "ROC-AUC (OvR)": m["roc_auc_weighted"],
            "Inference Time (s)": m["inference_time"],
        })

    df_comp = pd.DataFrame(records)

    print("\n" + "=" * 85)
    print(" SIDE-BY-SIDE CLASSICAL MODEL EVALUATION METRICS COMPARISON ")
    print("=" * 85)

    header = f"| {'Model':<22} | {'Macro F1':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<10} | {'Inf Time (s)':<12} |"
    divider = "+" + "-" * 105 + "+"
    print(divider)
    print(header)
    print(divider)

    for idx, row in df_comp.iterrows():
        print(
            f"| {row['Model']:<22} | {row['Macro F1']:<10.4f} | {row['Accuracy']:<10.4f} | "
            f"{row['Weighted Precision']:<10.4f} | {row['Weighted Recall']:<10.4f} | "
            f"{row['ROC-AUC (OvR)']:<10.4f} | {row['Inference Time (s)']:<12.4f} |"
        )
    print(divider)

    # Determine best model based on Macro F1
    best_model_metric = max(metrics_list, key=lambda x: x["macro_f1"])
    best_name = best_model_metric["model_name"]
    best_f1 = best_model_metric["macro_f1"]
    best_acc = best_model_metric["accuracy"]

    print("\n[MODEL SELECTION JUSTIFICATION (Primary Metric: Macro F1)]")
    print(f"  - Best Model : {best_name}")
    print(f"  - Macro F1   : {best_f1:.4f}")
    print(f"  - Accuracy   : {best_acc:.4f}")
    print(
        f"  - Rationale  : {best_name} achieves the highest Macro F1 score ({best_f1:.4f}) on the held-out "
        f"test set. Macro F1 treats all sentiment classes (Positive, Neutral, Negative) equally, preventing "
        f"the evaluation from being dominated by the majority Positive class while rewarding balanced multi-class performance."
    )
    print("=" * 85 + "\n")

    return df_comp


def run_evaluation_pipeline(
    data_dir: str = "outputs/cleaned_data",
    model_dir: str = "outputs/models",
    pred_dir: str = "outputs/predictions",
    fig_dir: str = "outputs/figures",
) -> Dict[str, Any]:
    """
    Execute full evaluation pipeline for classical models on test set:
      1. Load test split CSV (`outputs/cleaned_data/test.csv`).
      2. Load saved models & TF-IDF vectorizer (`outputs/models/`).
      3. Transform test text using vectorizer.
      4. Evaluate Logistic Regression on test set.
      5. Evaluate Linear SVM on test set.
      6. Save predictions CSVs & evaluation artifacts (confusion matrix plots, classification reports).
      7. Print side-by-side comparison table & model selection rationale.

    Returns:
        Dict[str, Any]: Combined evaluation results and comparison DataFrame.
    """
    print("=" * 70)
    print(" Classical Machine Learning Test Set Evaluation Pipeline ")
    print("=" * 70)

    # 1. Load Test Dataset
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split CSV not found at '{test_path}'. Run src/preprocessing.py or src/sentiment.py first.")

    df_test = pd.read_csv(test_path)
    X_test_text = df_test["cleaned_review"].astype(str)
    y_test = df_test["label"].values

    print(f"Loaded test dataset: {len(df_test):,} samples from '{test_path}'.")

    # 2. Load Saved Vectorizer and Models
    lr_path = os.path.join(model_dir, "logistic_regression.joblib")
    svm_path = os.path.join(model_dir, "linear_svm.joblib")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

    vectorizer = load_model(vec_path, model_type="sklearn")
    lr_model = load_model(lr_path, model_type="sklearn")
    svm_model = load_model(svm_path, model_type="sklearn")

    # 3. Transform Test Features
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"Transformed test features with TF-IDF: shape {X_test_tfidf.shape}.")

    # 4. Evaluate Logistic Regression
    lr_metrics = evaluate_sklearn_model(lr_model, X_test_tfidf, y_test, model_name="Logistic Regression")
    save_predictions(df_test, lr_metrics["y_pred"], "Logistic Regression", pred_dir)
    save_evaluation_artifacts(lr_metrics, pred_dir, fig_dir)

    # 5. Evaluate Linear SVM
    svm_metrics = evaluate_sklearn_model(svm_model, X_test_tfidf, y_test, model_name="Linear SVM")
    save_predictions(df_test, svm_metrics["y_pred"], "Linear SVM", pred_dir)
    save_evaluation_artifacts(svm_metrics, pred_dir, fig_dir)

    # 6. Side-by-side comparison & justification
    metrics_list = [lr_metrics, svm_metrics]
    df_comparison = compare_models(metrics_list)

    return {
        "lr_metrics": lr_metrics,
        "svm_metrics": svm_metrics,
        "comparison_table": df_comparison,
    }


if __name__ == "__main__":
    run_evaluation_pipeline()
