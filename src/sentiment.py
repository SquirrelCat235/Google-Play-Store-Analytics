"""
Sentiment Analysis Classical Machine Learning Training Module
================================================================
Implements classical ML model training pipelines for review sentiment classification.
Includes:
  - Logistic Regression with class weighting and GridSearchCV hyperparameter tuning.
  - Linear SVM (LinearSVC) with class weighting and GridSearchCV hyperparameter tuning.
  - Model and vectorizer persistence to outputs/models/.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports
from src.utils import set_seed, save_model, format_time
from src.preprocessing import load_reviews, preprocess_dataset, get_tfidf_features


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

    Args:
        X_train: Training feature matrix (TF-IDF).
        y_train: Training labels.
        param_grid (Dict[str, Any], optional): Hyperparameter grid for tuning.
        cv (int): Number of cross-validation folds (default 5).
        scoring (str): Cross-validation evaluation metric (default 'f1_weighted').
        random_state (int): Random seed for reproducibility (default 42).
        n_jobs (int): Number of parallel CPU jobs for grid search (default -1).

    Returns:
        Tuple[LogisticRegression, Dict[str, Any]]: (best_trained_model, metadata_dict)
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

    print("\n--- Training Logistic Regression (GridSearchCV 5-Fold) ---")
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

    print(f"Logistic Regression Training Complete in {format_time(training_time)} ({training_time:.2f}s):")
    print(f"  Best Parameters : {metadata['best_params']}")
    print(f"  Best CV Score   : {metadata['best_score']:.4f} ({scoring})")

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

    Args:
        X_train: Training feature matrix (TF-IDF).
        y_train: Training labels.
        param_grid (Dict[str, Any], optional): Hyperparameter grid for tuning.
        cv (int): Number of cross-validation folds (default 5).
        scoring (str): Cross-validation evaluation metric (default 'f1_weighted').
        random_state (int): Random seed for reproducibility (default 42).
        n_jobs (int): Number of parallel CPU jobs for grid search (default -1).

    Returns:
        Tuple[LinearSVC, Dict[str, Any]]: (best_trained_model, metadata_dict)
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

    print("\n--- Training Linear SVM (GridSearchCV 5-Fold) ---")
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

    print(f"Linear SVM Training Complete in {format_time(training_time)} ({training_time:.2f}s):")
    print(f"  Best Parameters : {metadata['best_params']}")
    print(f"  Best CV Score   : {metadata['best_score']:.4f} ({scoring})")

    return best_model, metadata


def run_classical_training_pipeline(
    data_path: str = "data/googleplaystore_user_reviews.csv",
    output_model_dir: str = "outputs/models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute full classical training pipeline:
      1. Set seed & load reviews dataset.
      2. Clean text & perform stratified train/val/test split.
      3. Extract TF-IDF features from training text.
      4. Train Logistic Regression with 5-fold CV hyperparameter tuning.
      5. Train Linear SVM with 5-fold CV hyperparameter tuning.
      6. Save Logistic Regression model, Linear SVM model, and TF-IDF vectorizer.
      7. Print detailed execution summary & comparison table.

    Returns:
        Dict[str, Any]: Dictionary containing trained models, vectorizer, and metadata.
    """
    set_seed(random_state)
    print("=" * 65)
    print(" Classical Machine Learning Training Pipeline ")
    print("=" * 65)

    # 1. Load Data
    raw_df = load_reviews(data_path)

    # 2. Preprocess & Split
    full_df, train_df, val_df, test_df, label2id, id2label = preprocess_dataset(
        raw_df, random_state=random_state
    )

    # 3. Extract TF-IDF Features
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(
        X_train=train_df["cleaned_review"],
        X_val=val_df["cleaned_review"],
        X_test=test_df["cleaned_review"],
        max_features=20000,
        ngram_range=(1, 2),
    )
    y_train = train_df["label"].values

    # 4. Train Logistic Regression
    lr_model, lr_meta = train_logistic_regression(
        X_train=X_train_tfidf,
        y_train=y_train,
        random_state=random_state,
    )

    # 5. Train Linear SVM
    svm_model, svm_meta = train_linear_svm(
        X_train=X_train_tfidf,
        y_train=y_train,
        random_state=random_state,
    )

    # 6. Save Artifacts using utils.save_model()
    os.makedirs(output_model_dir, exist_ok=True)
    lr_path = os.path.join(output_model_dir, "logistic_regression.joblib")
    svm_path = os.path.join(output_model_dir, "linear_svm.joblib")
    vec_path = os.path.join(output_model_dir, "tfidf_vectorizer.joblib")

    save_model(lr_model, lr_path, model_type="sklearn")
    save_model(svm_model, svm_path, model_type="sklearn")
    save_model(vectorizer, vec_path, model_type="sklearn")

    # 7. Print Pipeline Results Summary
    num_features = len(vectorizer.vocabulary_)
    print("\n" + "=" * 65)
    print(" TRAINING PIPELINE EXECUTION RESULTS ")
    print("=" * 65)
    print(f"Number of TF-IDF Features : {num_features:,}")
    print(f"Model Save Directory     : {output_model_dir}")
    print(f"  - Logistic Regression   : {lr_path}")
    print(f"  - Linear SVM            : {svm_path}")
    print(f"  - TF-IDF Vectorizer     : {vec_path}\n")

    # 8. Print Comparison Table
    print("+" + "-" * 75 + "+")
    print(f"| {'Model':<22} | {'Best Parameters':<22} | {'Best CV Score':<13} | {'Training Time':<10} |")
    print("+" + "-" * 75 + "+")

    lr_param_str = str(lr_meta['best_params']).replace(" ", "")
    svm_param_str = str(svm_meta['best_params']).replace(" ", "")

    print(f"| {'Logistic Regression':<22} | {lr_param_str:<22} | {lr_meta['best_score']:<13.4f} | {format_time(lr_meta['training_time']):<10} |")
    print(f"| {'Linear SVM':<22} | {svm_param_str:<22} | {svm_meta['best_score']:<13.4f} | {format_time(svm_meta['training_time']):<10} |")
    print("+" + "-" * 75 + "+")

    return {
        "logistic_regression": lr_model,
        "linear_svm": svm_model,
        "vectorizer": vectorizer,
        "lr_metadata": lr_meta,
        "svm_metadata": svm_meta,
        "save_paths": {
            "logistic_regression": lr_path,
            "linear_svm": svm_path,
            "vectorizer": vec_path,
        },
    }


if __name__ == "__main__":
    run_classical_training_pipeline()
