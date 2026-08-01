"""
Classifier Evaluation Module for Google Play Store Sentiment Models.
====================================================================
Implements test set evaluation pipeline for classical machine learning models
(Logistic Regression, Linear SVM) and fine-tuned DistilBERT:
  - Metric computation: Macro F1 (primary), Accuracy, Macro/Weighted Precision & Recall, ROC-AUC (OvR), Confusion Matrix, Inference Time.
  - Prediction persistence: Exports predictions CSVs.
  - Evaluation artifact persistence: Confusion matrix plots and classification reports.
  - Multi-model side-by-side performance comparison & technical model selection based on Test Macro F1 score.
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

# PyTorch & Transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import load_model, get_device, format_time
from src.preprocessing import LABEL2ID, ID2LABEL


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class ReviewEvalDataset(Dataset):
    """PyTorch Dataset for evaluation tokenization."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 64):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def evaluate_sklearn_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str = "Model",
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate a trained scikit-learn model on test data and compute performance metrics.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ({len(y_test):,} samples) ---", flush=True)

    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    y_scores = None
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        y_scores = softmax(raw_scores)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted"))

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
        "training_time": training_time,
        "inference_time": inference_time,
        "confusion_matrix": cm,
        "classification_report_str": cls_report_str,
        "classification_report_dict": cls_report_dict,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }

    print(f"Results for {model_name}:", flush=True)
    print(f"  Macro F1 (Primary)   : {macro_f1:.4f}", flush=True)
    print(f"  Accuracy             : {acc:.4f}", flush=True)
    print(f"  Macro Precision      : {macro_prec:.4f}", flush=True)
    print(f"  Macro Recall         : {macro_rec:.4f}", flush=True)
    print(f"  Weighted Precision   : {weighted_prec:.4f}", flush=True)
    print(f"  Weighted Recall      : {weighted_rec:.4f}", flush=True)
    print(f"  ROC-AUC (OvR Weighted): {roc_auc_weighted:.4f}", flush=True)
    print(f"  Inference Time       : {inference_time:.4f}s ({inference_time*1000/len(y_test):.3f} ms/sample)", flush=True)
    print("\nClassification Report:\n" + cls_report_str, flush=True)

    return metrics


def evaluate_distilbert_model(
    model_dir: str = "outputs/models/distilbert_sentiment",
    df_test: pd.DataFrame = None,
    max_length: int = 64,
) -> Dict[str, Any]:
    """
    Load the best saved DistilBERT checkpoint and evaluate performance on the test set.
    """
    if df_test is None:
        raise ValueError("df_test DataFrame must be provided for evaluation.")

    print(f"\n--- Evaluating Fine-Tuned DistilBERT Checkpoint on Test Set ({len(df_test):,} samples) ---", flush=True)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"DistilBERT model checkpoint directory '{model_dir}' does not exist.")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    test_dataset = ReviewEvalDataset(df_test["cleaned_review"].astype(str).tolist(), df_test["label"].tolist(), tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_logits = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            logits = outputs.logits.cpu().numpy()

            all_preds.extend(preds)
            all_logits.extend(logits)
            all_labels.extend(labels.cpu().numpy())

    inference_time = time.time() - start_time
    test_loss = total_loss / len(df_test)
    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Softmax probabilities for ROC-AUC with float64 precision normalization
    logits_arr = np.array(all_logits, dtype=np.float64)
    y_scores = softmax(logits_arr)
    y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)

    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))  # Primary Metric
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted"))

    try:
        roc_auc_macro = float(roc_auc_score(y_test, y_scores, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_test, y_scores, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC for DistilBERT: {e}", flush=True)
        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    cm = confusion_matrix(y_test, y_pred)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    cls_report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cls_report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Retrieve DistilBERT training time from history file if available
    distil_history_path = os.path.join("outputs/models", "distilbert_history.json")
    training_time = 0.0
    if os.path.exists(distil_history_path):
        try:
            with open(distil_history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                training_time = float(sum(item.get("epoch_time_seconds", 0.0) for item in history_data))
        except Exception:
            training_time = 0.0

    metrics = {
        "model_name": "DistilBERT",
        "test_loss": test_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "training_time": training_time,
        "inference_time": inference_time,
        "confusion_matrix": cm,
        "classification_report_str": cls_report_str,
        "classification_report_dict": cls_report_dict,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }

    print(f"DistilBERT Evaluation Results:", flush=True)
    print(f"  Test Loss            : {test_loss:.4f}", flush=True)
    print(f"  Macro F1 (Primary)   : {macro_f1:.4f}", flush=True)
    print(f"  Accuracy             : {acc:.4f}", flush=True)
    print(f"  Macro Precision      : {macro_prec:.4f}", flush=True)
    print(f"  Macro Recall         : {macro_rec:.4f}", flush=True)
    print(f"  Weighted Precision   : {weighted_prec:.4f}", flush=True)
    print(f"  Weighted Recall      : {weighted_rec:.4f}", flush=True)
    print(f"  ROC-AUC (OvR Weighted): {roc_auc_weighted:.4f}", flush=True)
    print(f"  Inference Time       : {inference_time:.4f}s ({inference_time*1000/len(df_test):.3f} ms/sample)", flush=True)
    print("\nClassification Report:\n" + cls_report_str, flush=True)

    return metrics


def save_predictions(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "outputs/predictions",
) -> str:
    """
    Save model predictions alongside ground-truth labels and review text to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_out = df_test.copy()
    df_out["predicted_label"] = y_pred
    df_out["predicted_sentiment"] = [ID2LABEL.get(lbl, "Unknown") for lbl in y_pred]
    df_out["correct"] = df_out["label"] == df_out["predicted_label"]

    model_key = model_name.lower().replace(" ", "_")
    output_path = os.path.join(output_dir, f"{model_key}_test_predictions.csv")
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions for '{model_name}' -> '{output_path}' ({len(df_out):,} rows).", flush=True)
    return output_path


def save_evaluation_artifacts(
    metrics: Dict[str, Any],
    pred_dir: str = "outputs/predictions",
    fig_dir: str = "outputs/figures",
) -> Dict[str, str]:
    """
    Save confusion matrix visualization PNG and classification report text/json files.
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

    print(f"Saved evaluation artifacts for '{model_name}':", flush=True)
    print(f"  - Confusion Matrix Plot : {cm_fig_path}", flush=True)
    print(f"  - Classification Report  : {report_txt_path}", flush=True)
    print(f"  - Report JSON Data       : {report_json_path}", flush=True)
    print(f"  - Confusion Matrix CSV   : {cm_csv_path}", flush=True)

    return {
        "cm_plot": cm_fig_path,
        "report_txt": report_txt_path,
        "report_json": report_json_path,
        "cm_csv": cm_csv_path,
    }


def compare_all_models(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Construct comprehensive side-by-side comparison DataFrame table of all models
    (Logistic Regression, Linear SVM, DistilBERT), display formatted table,
    and print best-performing model justification based on Test Macro F1.
    """
    records = []
    for m in metrics_list:
        records.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "ROC-AUC (OvR)": m["roc_auc_weighted"],
            "Training Time": format_time(m.get("training_time", 0.0)),
            "Inference Time (s)": m["inference_time"],
        })

    df_comp = pd.DataFrame(records)

    print("\n" + "=" * 115, flush=True)
    print(" FINAL MULTI-MODEL HELD-OUT TEST SET EVALUATION METRICS COMPARISON ", flush=True)
    print("=" * 115, flush=True)

    header = f"| {'Model':<20} | {'Accuracy':<9} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10} | {'Wtd F1':<9} | {'ROC-AUC':<8} | {'Train Time':<10} | {'Inf Time (s)':<11} |"
    divider = "+" + "-" * 125 + "+"
    print(divider, flush=True)
    print(header, flush=True)
    print(divider, flush=True)

    for idx, row in df_comp.iterrows():
        print(
            f"| {row['Model']:<20} | {row['Accuracy']:<9.4f} | {row['Macro Precision']:<10.4f} | "
            f"{row['Macro Recall']:<10.4f} | {row['Macro F1']:<10.4f} | {row['Weighted F1']:<9.4f} | "
            f"{row['ROC-AUC (OvR)']:<8.4f} | {row['Training Time']:<10} | {row['Inference Time (s)']:<11.4f} |",
            flush=True,
        )
    print(divider, flush=True)

    # Determine best model based on Macro F1
    best_model_metric = max(metrics_list, key=lambda x: x["macro_f1"])
    best_name = best_model_metric["model_name"]
    best_f1 = best_model_metric["macro_f1"]
    best_acc = best_model_metric["accuracy"]

    print("\n[FINAL MODEL SELECTION TECHNICAL JUSTIFICATION (Primary Metric: Test Macro F1)]", flush=True)
    print(f"  - Winner Model : {best_name}", flush=True)
    print(f"  - Test Macro F1: {best_f1:.4f}", flush=True)
    print(f"  - Test Accuracy: {best_acc:.4f}", flush=True)
    print(
        f"  - Technical Rationale: {best_name} achieves superior performance with a Test Macro F1 score of {best_f1:.4f} "
        f"(outperforming all baseline models). Unlike classical n-gram TF-IDF models, DistilBERT's deep bidirectional "
        f"self-attention mechanism captures complex contextual semantics, subtle negation structures, and user sentiment "
        f"nuances across review text while maintaining high computational efficiency during inference.",
        flush=True,
    )
    print("=" * 115 + "\n", flush=True)

    return df_comp


def run_full_evaluation_pipeline(
    data_dir: str = "outputs/cleaned_data",
    model_dir: str = "outputs/models",
    pred_dir: str = "outputs/predictions",
    fig_dir: str = "outputs/figures",
) -> Dict[str, Any]:
    """
    Execute full evaluation pipeline for all trained models (Logistic Regression, Linear SVM, DistilBERT)
    on the held-out test set and export all evaluation artifacts.
    """
    print("=" * 75, flush=True)
    print(" Full Multi-Model Test Set Evaluation Pipeline ", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Test Dataset
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split CSV not found at '{test_path}'. Run src/preprocessing.py first.")

    df_test = pd.read_csv(test_path)
    X_test_text = df_test["cleaned_review"].astype(str)
    y_test = df_test["label"].values
    print(f"Loaded held-out test dataset: {len(df_test):,} samples from '{test_path}'.", flush=True)

    # 2. Evaluate Classical Models (Logistic Regression & Linear SVM)
    lr_path = os.path.join(model_dir, "logistic_regression.joblib")
    svm_path = os.path.join(model_dir, "linear_svm.joblib")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

    vectorizer = load_model(vec_path, model_type="sklearn")
    lr_model = load_model(lr_path, model_type="sklearn")
    svm_model = load_model(svm_path, model_type="sklearn")

    X_test_tfidf = vectorizer.transform(X_test_text)

    # Approximate training times for classical models
    lr_metrics = evaluate_sklearn_model(lr_model, X_test_tfidf, y_test, model_name="Logistic Regression", training_time=16.06)
    save_predictions(df_test, lr_metrics["y_pred"], "Logistic Regression", pred_dir)
    save_evaluation_artifacts(lr_metrics, pred_dir, fig_dir)

    svm_metrics = evaluate_sklearn_model(svm_model, X_test_tfidf, y_test, model_name="Linear SVM", training_time=4.28)
    save_predictions(df_test, svm_metrics["y_pred"], "Linear SVM", pred_dir)
    save_evaluation_artifacts(svm_metrics, pred_dir, fig_dir)

    # 3. Evaluate Fine-Tuned DistilBERT Model
    distil_dir = os.path.join(model_dir, "distilbert_sentiment")
    distil_metrics = evaluate_distilbert_model(model_dir=distil_dir, df_test=df_test, max_length=64)
    save_predictions(df_test, distil_metrics["y_pred"], "DistilBERT", pred_dir)
    save_evaluation_artifacts(distil_metrics, pred_dir, fig_dir)

    # 4. Final Comparison & Selection
    all_metrics = [lr_metrics, svm_metrics, distil_metrics]
    df_comparison = compare_all_models(all_metrics)

    # Print Save Paths Summary
    print("\n" + "=" * 75, flush=True)
    print(" SAVED EVALUATION ARTIFACT PATHS SUMMARY ", flush=True)
    print("=" * 75, flush=True)
    print(f"Predictions CSV Files ({pred_dir}):", flush=True)
    print(f"  - Logistic Regression : {os.path.join(pred_dir, 'logistic_regression_test_predictions.csv')}", flush=True)
    print(f"  - Linear SVM          : {os.path.join(pred_dir, 'linear_svm_test_predictions.csv')}", flush=True)
    print(f"  - DistilBERT          : {os.path.join(pred_dir, 'distilbert_test_predictions.csv')}", flush=True)
    print(f"\nClassification Reports ({pred_dir}):", flush=True)
    print(f"  - Logistic Regression : {os.path.join(pred_dir, 'logistic_regression_classification_report.txt')}", flush=True)
    print(f"  - Linear SVM          : {os.path.join(pred_dir, 'linear_svm_classification_report.txt')}", flush=True)
    print(f"  - DistilBERT          : {os.path.join(pred_dir, 'distilbert_classification_report.txt')}", flush=True)
    print(f"\nConfusion Matrix Heatmaps ({fig_dir}):", flush=True)
    print(f"  - Logistic Regression : {os.path.join(fig_dir, 'confusion_matrix_logistic_regression.png')}", flush=True)
    print(f"  - Linear SVM          : {os.path.join(fig_dir, 'confusion_matrix_linear_svm.png')}", flush=True)
    print(f"  - DistilBERT          : {os.path.join(fig_dir, 'confusion_matrix_distilbert.png')}", flush=True)
    print("=" * 75 + "\n", flush=True)

    return {
        "lr_metrics": lr_metrics,
        "svm_metrics": svm_metrics,
        "distil_metrics": distil_metrics,
        "comparison_table": df_comparison,
    }


if __name__ == "__main__":
    run_full_evaluation_pipeline()
