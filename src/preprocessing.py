"""
Text preprocessing and dataset preparation module for Google Play Store NLP Analytics.
Handles loading reviews, text cleaning, categorical label encoding, stratified splitting,
TF-IDF feature extraction, and exporting cleaned datasets.
"""

import os
import re
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

LABEL2ID: Dict[str, int] = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def load_reviews(path: str = "data/googleplaystore_user_reviews.csv") -> pd.DataFrame:
    """
    Load the Google Play reviews CSV, retain rows with non-null review text and sentiment labels,
    remove exact duplicates, and return the cleaned DataFrame.

    Args:
        path (str): Path to the user reviews CSV file.

    Returns:
        pd.DataFrame: DataFrame containing valid labeled reviews.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Review CSV not found at '{path}'.")

    df = pd.read_csv(path)

    # Required columns in the dataset
    review_col = "Translated_Review"
    sentiment_col = "Sentiment"

    if review_col not in df.columns or sentiment_col not in df.columns:
        raise KeyError(f"Expected columns '{review_col}' and '{sentiment_col}' in dataset.")

    # Keep rows with non-null review text and non-null sentiment
    df_clean = df.dropna(subset=[review_col, sentiment_col]).copy()

    # Ensure sentiment contains only expected classes
    df_clean = df_clean[df_clean[sentiment_col].isin(LABEL2ID.keys())].copy()

    # Deduplicate based on Translated_Review and Sentiment
    df_clean = df_clean.drop_duplicates(subset=[review_col, sentiment_col]).reset_index(drop=True)

    print(f"Loaded {len(df_clean):,} valid labeled reviews after dropping missing values and duplicates.")
    return df_clean


def clean_text(text: Any) -> str:
    """
    Clean review text by lowercasing, removing URLs, HTML tags, special characters,
    collapsing multiple spaces, and stripping leading/trailing whitespace.
    Gracefully handles missing or non-string inputs.

    Args:
        text (Any): Input text or value.

    Returns:
        str: Cleaned text string.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove non-alphanumeric characters except whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    output_dir: str = "outputs/cleaned_data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Clean review texts, encode sentiment labels (Negative=0, Neutral=1, Positive=2),
    perform stratified train/val/test split (70/15/15), export splits to CSV,
    and return the full labeled DataFrame along with train, val, and test splits.

    Args:
        df (pd.DataFrame): Raw labeled DataFrame from load_reviews().
        test_size (float): Proportion of dataset for test split (default 0.15).
        val_size (float): Proportion of dataset for validation split (default 0.15).
        random_state (int): Random seed for reproducibility (default 42).
        output_dir (str): Directory where train/val/test CSVs will be saved.

    Returns:
        Tuple containing (full_df, train_df, val_df, test_df, label2id, id2label)
    """
    df = df.copy()

    # Apply text cleaning
    df["cleaned_review"] = df["Translated_Review"].apply(clean_text)

    # Filter out any empty reviews that became blank after cleaning
    df = df[df["cleaned_review"].str.len() > 0].reset_index(drop=True)

    # Encode sentiment labels
    df["label"] = df["Sentiment"].map(LABEL2ID)

    # First split: Separate train + val (85%) and test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Second split: Separate train (70% overall) and val (15% overall)
    # Relative val size within train_val is 0.15 / 0.85 ≈ 0.17647
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save cleaned splits to output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset preprocessed and split successfully:")
    print(f"  Full Labeled Data : {len(df):,} samples")
    print(f"  Train Split (70%) : {len(train_df):,} samples -> saved to '{train_path}'")
    print(f"  Val Split   (15%) : {len(val_df):,} samples -> saved to '{val_path}'")
    print(f"  Test Split  (15%) : {len(test_df):,} samples -> saved to '{test_path}'")

    return df, train_df, val_df, test_df, LABEL2ID, ID2LABEL


def get_tfidf_features(
    X_train: pd.Series,
    X_val: pd.Series,
    X_test: pd.Series,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[Any, Any, Any, TfidfVectorizer]:
    """
    Extract TF-IDF features from text series. Vectorizer is fit exclusively on training data
    to prevent data leakage, then transforms validation and test data.

    Args:
        X_train (pd.Series): Cleaned review text for training.
        X_val (pd.Series): Cleaned review text for validation.
        X_test (pd.Series): Cleaned review text for testing.
        max_features (int): Maximum vocabulary size (default 20,000).
        ngram_range (Tuple[int, int]): Range of n-grams (default (1, 2)).

    Returns:
        Tuple containing (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Feature Extraction Complete:")
    print(f"  Vocabulary Size : {len(vectorizer.vocabulary_):,} features")
    print(f"  X_train TF-IDF  : {X_train_tfidf.shape}")
    print(f"  X_val TF-IDF    : {X_val_tfidf.shape}")
    print(f"  X_test TF-IDF   : {X_test_tfidf.shape}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
