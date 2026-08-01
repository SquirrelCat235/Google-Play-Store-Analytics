"""
Sentiment Analysis Machine Learning Training Module
================================================================
Implements classical ML (Logistic Regression, Linear SVM) and Fine-Tuned DistilBERT
training pipelines for Google Play Store review sentiment classification.
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress minor warnings for clean attached console output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# PyTorch & Transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports
from src.utils import set_seed, get_device, save_model, format_time
from src.preprocessing import load_reviews, preprocess_dataset, get_tfidf_features, LABEL2ID, ID2LABEL


# Enable PyTorch CUDNN benchmarks if CUDA is active
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# =====================================================================
# CLASSICAL MACHINE LEARNING PIPELINES
# =====================================================================

def train_logistic_regression(
    X_train: Any,
    y_train: Any,
    param_grid: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a Logistic Regression model with balanced class weights and 5-fold GridSearchCV.
    """
    if param_grid is None:
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        }

    base_model = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=2000,
    )

    print("\n--- Training Logistic Regression (GridSearchCV 5-Fold) ---", flush=True)
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    metadata = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "training_time": training_time,
        "cv_folds": cv,
        "scoring_metric": scoring,
    }

    print(f"Logistic Regression Training Complete in {format_time(training_time)} ({training_time:.2f}s):", flush=True)
    print(f"  Best Parameters : {metadata['best_params']}", flush=True)
    print(f"  Best CV Score   : {metadata['best_score']:.4f} ({scoring})", flush=True)

    return best_model, metadata


def train_linear_svm(
    X_train: Any,
    y_train: Any,
    param_grid: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[LinearSVC, Dict[str, Any]]:
    """
    Train a Linear Support Vector Machine (LinearSVC) with balanced class weights and 5-fold GridSearchCV.
    """
    if param_grid is None:
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
        }

    base_model = LinearSVC(
        class_weight="balanced",
        random_state=random_state,
        max_iter=3000,
        dual="auto",
    )

    print("\n--- Training Linear SVM (GridSearchCV 5-Fold) ---", flush=True)
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    metadata = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "training_time": training_time,
        "cv_folds": cv,
        "scoring_metric": scoring,
    }

    print(f"Linear SVM Training Complete in {format_time(training_time)} ({training_time:.2f}s):", flush=True)
    print(f"  Best Parameters : {metadata['best_params']}", flush=True)
    print(f"  Best CV Score   : {metadata['best_score']:.4f} ({scoring})", flush=True)

    return best_model, metadata


def run_classical_training_pipeline(
    data_path: str = "data/googleplaystore_user_reviews.csv",
    output_model_dir: str = "outputs/models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute full classical training pipeline.
    """
    set_seed(random_state)
    print("=" * 65, flush=True)
    print(" Classical Machine Learning Training Pipeline ", flush=True)
    print("=" * 65, flush=True)

    raw_df = load_reviews(data_path)
    full_df, train_df, val_df, test_df, label2id, id2label = preprocess_dataset(
        raw_df, random_state=random_state
    )

    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(
        X_train=train_df["cleaned_review"],
        X_val=val_df["cleaned_review"],
        X_test=test_df["cleaned_review"],
        max_features=20000,
        ngram_range=(1, 2),
    )
    y_train = train_df["label"].values

    lr_model, lr_meta = train_logistic_regression(X_train=X_train_tfidf, y_train=y_train, random_state=random_state)
    svm_model, svm_meta = train_linear_svm(X_train=X_train_tfidf, y_train=y_train, random_state=random_state)

    os.makedirs(output_model_dir, exist_ok=True)
    lr_path = os.path.join(output_model_dir, "logistic_regression.joblib")
    svm_path = os.path.join(output_model_dir, "linear_svm.joblib")
    vec_path = os.path.join(output_model_dir, "tfidf_vectorizer.joblib")

    save_model(lr_model, lr_path, model_type="sklearn")
    save_model(svm_model, svm_path, model_type="sklearn")
    save_model(vectorizer, vec_path, model_type="sklearn")

    num_features = len(vectorizer.vocabulary_)
    print("\n" + "=" * 65, flush=True)
    print(" TRAINING PIPELINE EXECUTION RESULTS ", flush=True)
    print("=" * 65, flush=True)
    print(f"Number of TF-IDF Features : {num_features:,}", flush=True)
    print(f"Model Save Directory     : {output_model_dir}\n", flush=True)

    return {
        "logistic_regression": lr_model,
        "linear_svm": svm_model,
        "vectorizer": vectorizer,
        "lr_metadata": lr_meta,
        "svm_metadata": svm_meta,
    }


# =====================================================================
# DISTILBERT TRANSFORMER TRAINING PIPELINE
# =====================================================================

class ReviewDataset(Dataset):
    """
    PyTorch Dataset for Transformer tokenized review texts and sentiment labels.
    """
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


def analyze_token_lengths(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    target_coverage: float = 0.95,
) -> Dict[str, Any]:
    """
    Analyze the token length distribution of review texts using the specified tokenizer
    and select an optimal maximum sequence length.
    """
    print("\n--- Analyzing Review Token Length Distribution ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    token_lengths = [len(toks) for toks in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    token_lengths = np.array(token_lengths)

    mean_len = float(np.mean(token_lengths))
    median_len = float(np.median(token_lengths))
    p95_len = float(np.percentile(token_lengths, 95))
    max_len = int(np.max(token_lengths))

    if p95_len <= 64:
        selected_max_length = 64
    elif p95_len <= 128:
        selected_max_length = 128
    else:
        selected_max_length = 256

    coverage_pct = float((token_lengths <= selected_max_length).mean() * 100)

    print(f"Token Length Analysis Results ({len(texts):,} training samples):", flush=True)
    print(f"  - Average Token Length  : {mean_len:.2f} tokens", flush=True)
    print(f"  - Median Token Length   : {median_len:.0f} tokens", flush=True)
    print(f"  - 95th Percentile Length: {p95_len:.0f} tokens", flush=True)
    print(f"  - Maximum Token Length  : {max_len} tokens", flush=True)
    print(f"  - Selected max_length   : {selected_max_length} tokens", flush=True)
    print(f"  - Coverage w/o Truncation: {coverage_pct:.2f}%", flush=True)
    print(
        f"  - Justification         : A max_length of {selected_max_length} covers {coverage_pct:.2f}% of all "
        f"reviews without truncation while maximizing computational efficiency and memory utilization.",
        flush=True,
    )

    return {
        "mean_length": mean_len,
        "median_length": median_len,
        "p95_length": p95_len,
        "max_length_stat": max_len,
        "selected_max_length": selected_max_length,
        "coverage_percentage": coverage_pct,
    }


def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    lr: float = 2e-5,
    patience: int = 1,
    weight_decay: float = 0.01,
    random_state: int = 42,
    output_dir: str = "outputs/models/distilbert_sentiment",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Train and fine-tune DistilBERT model with weighted CrossEntropyLoss, AdamW optimizer,
    linear learning rate scheduler with warmup, early stopping based on Validation Macro F1,
    and automatic GPU/AMP hardware configuration.
    """
    set_seed(random_state)
    print("\n" + "=" * 70, flush=True)
    print(" Fine-Tuning DistilBERT Model (distilbert-base-uncased) ")
    print("=" * 70, flush=True)

    # 1. Device and Hardware Configuration
    device = get_device()
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        precision_mode = "BF16" if bf16_supported else "FP16"
        batch_size = 16  # Optimized for RTX 2050 (4 GB VRAM) to prevent OOM
        grad_accum_steps = 2  # Effective batch size = 32
    else:
        gpu_name = "N/A (CPU execution)"
        vram_gb = 0.0
        precision_mode = "FP32"
        batch_size = 32
        grad_accum_steps = 1

    print(f"\n[Execution Configuration]", flush=True)
    print(f"  Device              : {device}", flush=True)
    print(f"  GPU Name            : {gpu_name}", flush=True)
    print(f"  Available VRAM      : {vram_gb:.2f} GB" if is_cuda else "  Available VRAM      : N/A", flush=True)
    print(f"  Precision Mode      : {precision_mode}", flush=True)
    print(f"  Per-Device Batch    : {batch_size}", flush=True)
    print(f"  Grad Accumulation   : {grad_accum_steps} (Effective Batch Size: {batch_size * grad_accum_steps})", flush=True)
    print(f"  Early Stopping      : Enabled (Patience = {patience} epoch(s) on Validation Macro F1)", flush=True)

    # 2. Token Length Analysis & Dataset Tokenization
    token_stats = analyze_token_lengths(train_df["cleaned_review"].tolist(), model_name=model_name)
    max_length = token_stats["selected_max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ReviewDataset(train_df["cleaned_review"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = ReviewDataset(val_df["cleaned_review"].tolist(), val_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=is_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=is_cuda)

    # 3. Model & Loss Function Setup (Weighted CrossEntropy)
    y_train = train_df["label"].values
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(f"\nComputed Class Weights for Loss Function: {class_weights_tensor.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # AMP Scaler for Mixed Precision
    use_amp = is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Training Loop with Early Stopping Monitoring
    print("\n" + "=" * 85, flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Macro F1':<12} | {'Val Acc':<9} | {'LR':<9} | {'Epoch Time':<10} | {'Est. Left':<9}", flush=True)
    print("=" * 85, flush=True)

    best_val_macro_f1 = -1.0
    best_model_state = None
    best_epoch = 0
    no_improvement_count = 0
    history = []
    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        # Batch Training Loop with Live TQDM Progress Bar
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [Train]",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        )

        for step, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if precision_mode == "BF16" else torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / grad_accum_steps
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) / grad_accum_steps
                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            running_loss += loss.item() * grad_accum_steps
            train_bar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        train_loss = running_loss / len(train_loader)

        # Validation Loop with Live TQDM Progress Bar
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{epochs} [Val]  ",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        )
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = float(accuracy_score(val_labels, val_preds))
        val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))
        current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr

        epoch_time = time.time() - epoch_start_time

        if epoch == 1:
            est_remaining_seconds = epoch_time * (epochs - 1)
            est_remaining_str = format_time(est_remaining_seconds)
        else:
            est_remaining_seconds = epoch_time * (epochs - epoch)
            est_remaining_str = format_time(est_remaining_seconds)

        print(
            f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {val_macro_f1:<12.4f} | "
            f"{val_acc:<9.4f} | {current_lr:<9.2e} | {format_time(epoch_time):<10} | {est_remaining_str:<9}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        })

        # Early Stopping Logic based on Validation Macro F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improvement_count = 0
            print(f"  --> Epoch {epoch}: Best Validation Macro F1 improved to {best_val_macro_f1:.4f}. Model checkpoint saved.", flush=True)
        else:
            no_improvement_count += 1
            print(
                f"  --> Epoch {epoch}: Validation Macro F1 ({val_macro_f1:.4f}) did not improve over best ({best_val_macro_f1:.4f}). "
                f"Patience counter: {no_improvement_count}/{patience}.",
                flush=True,
            )
            if no_improvement_count >= patience:
                print(
                    f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 did not improve for {patience} consecutive epoch(s). "
                    f"Stopping training early at Epoch {epoch}.",
                    flush=True,
                )
                break

    print("=" * 85, flush=True)
    print(f"\n[Training Completed in {format_time(time.time() - total_start_time)}]", flush=True)
    print(f"  - Best Epoch               : Epoch {best_epoch}", flush=True)
    print(f"  - Best Validation Macro F1 : {best_val_macro_f1:.4f}", flush=True)

    # 6. Restore Best Model Checkpoint
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Restored best model checkpoint weights from Epoch {best_epoch}.", flush=True)

    # 7. Save Model, Tokenizer, and Training History
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history_file = os.path.join("outputs/models", "distilbert_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"Saved DistilBERT model & tokenizer -> '{output_dir}'.", flush=True)
    print(f"Saved DistilBERT training history  -> '{history_file}'.", flush=True)

    metadata = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "token_stats": token_stats,
        "output_dir": output_dir,
        "history_file": history_file,
    }

    return model, tokenizer, metadata


def run_distilbert_training_pipeline(
    data_path: str = "data/googleplaystore_user_reviews.csv",
    output_dir: str = "outputs/models/distilbert_sentiment",
    epochs: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute DistilBERT fine-tuning pipeline on Google Play Store reviews dataset.
    """
    set_seed(random_state)
    raw_df = load_reviews(data_path)
    full_df, train_df, val_df, test_df, label2id, id2label = preprocess_dataset(raw_df, random_state=random_state)

    model, tokenizer, metadata = train_distilbert(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        random_state=random_state,
        output_dir=output_dir,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_distilbert_training_pipeline()
