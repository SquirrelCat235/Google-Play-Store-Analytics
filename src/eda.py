"""
Exploratory Data Analysis (EDA) module for Google Play Store Review Sentiment.
Generates publication-quality visualizations (sentiment distributions, review length histograms,
word clouds, and top n-grams) on the full labeled dataset and exports them to outputs/figures/.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

PALETTE: Dict[str, str] = {
    "Positive": "#2ecc71",
    "Neutral": "#3498db",
    "Negative": "#e74c3c",
}


def plot_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_col: str = "Sentiment",
    output_dir: str = "outputs/figures",
) -> plt.Figure:
    """
    Plot and save the class count and percentage distribution of sentiments with explicit legend and labels.

    Args:
        df (pd.DataFrame): Full labeled DataFrame.
        sentiment_col (str): Column name containing sentiment labels.
        output_dir (str): Output directory for saved plots.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = df[sentiment_col].value_counts()
    percentages = df[sentiment_col].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[PALETTE.get(s, "#7f8c8d") for s in counts.index],
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )

    # Annotate bar values (Count & %)
    for bar, sentiment in zip(bars, counts.index):
        height = bar.get_height()
        pct = percentages[sentiment]
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Sentiment Class Distribution (Full Labeled Dataset)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Total Number of Reviews", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, max(counts.values) * 1.18)

    # Explicit Legend
    legend_patches = [
        mpatches.Patch(color=PALETTE[s], label=f"{s} ({percentages[s]:.1f}%)")
        for s in counts.index if s in PALETTE
    ]
    ax.legend(handles=legend_patches, title="Sentiment Class & Share", loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "sentiment_distribution.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved sentiment distribution plot to '{output_path}'.")
    return fig


def plot_review_length_distribution(
    df: pd.DataFrame,
    text_col: str = "cleaned_review",
    sentiment_col: str = "Sentiment",
    output_dir: str = "outputs/figures",
) -> Dict[str, float]:
    """
    Calculate review length statistics and plot character/word length distributions with clear legends and labels.

    Args:
        df (pd.DataFrame): Full labeled DataFrame.
        text_col (str): Column name containing cleaned review text.
        sentiment_col (str): Column name containing sentiment labels.
        output_dir (str): Output directory for saved plots.

    Returns:
        Dict[str, float]: Character and word length statistics (mean, median, p95, max).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_temp = df.copy()
    df_temp["char_length"] = df_temp[text_col].astype(str).str.len()
    df_temp["word_length"] = df_temp[text_col].astype(str).str.split().str.len()

    stats: Dict[str, float] = {
        "char_mean": float(df_temp["char_length"].mean()),
        "char_median": float(df_temp["char_length"].median()),
        "char_p95": float(np.percentile(df_temp["char_length"], 95)),
        "char_max": float(df_temp["char_length"].max()),
        "word_mean": float(df_temp["word_length"].mean()),
        "word_median": float(df_temp["word_length"].median()),
        "word_p95": float(np.percentile(df_temp["word_length"], 95)),
        "word_max": float(df_temp["word_length"].max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Construct explicit legend handles (Classes + Reference Thresholds)
    sentiment_patches = [
        mpatches.Patch(color=PALETTE["Positive"], label="Positive Sentiment (Green)"),
        mpatches.Patch(color=PALETTE["Neutral"], label="Neutral Sentiment (Blue)"),
        mpatches.Patch(color=PALETTE["Negative"], label="Negative Sentiment (Red)"),
    ]

    # Plot 1: Word length distribution by sentiment
    sns.histplot(
        data=df_temp,
        x="word_length",
        hue=sentiment_col,
        palette=PALETTE,
        kde=True,
        bins=50,
        ax=axes[0],
        element="step",
        legend=False,
    )
    med_line_w = axes[0].axvline(stats["word_median"], color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Median ({stats['word_median']:.0f} words)")
    p95_line_w = axes[0].axvline(stats["word_p95"], color="#e67e22", linestyle=":", linewidth=2.0, label=f"95th Pct ({stats['word_p95']:.0f} words)")
    axes[0].set_title("Word Length Distribution by Sentiment Class", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Word Count per Review", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frequency (Number of Reviews)", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, 150)

    # Subplot 1 explicit legend
    handles_w = sentiment_patches + [med_line_w, p95_line_w]
    axes[0].legend(handles=handles_w, title="Sentiment Class & Thresholds", loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # Plot 2: Character length distribution by sentiment
    sns.histplot(
        data=df_temp,
        x="char_length",
        hue=sentiment_col,
        palette=PALETTE,
        kde=True,
        bins=50,
        ax=axes[1],
        element="step",
        legend=False,
    )
    med_line_c = axes[1].axvline(stats["char_median"], color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Median ({stats['char_median']:.0f} chars)")
    p95_line_c = axes[1].axvline(stats["char_p95"], color="#e67e22", linestyle=":", linewidth=2.0, label=f"95th Pct ({stats['char_p95']:.0f} chars)")
    axes[1].set_title("Character Length Distribution by Sentiment Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Character Count per Review", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Frequency (Number of Reviews)", fontsize=11, fontweight="bold")
    axes[1].set_xlim(0, 800)

    # Subplot 2 explicit legend
    handles_c = sentiment_patches + [med_line_c, p95_line_c]
    axes[1].legend(handles=handles_c, title="Sentiment Class & Thresholds", loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "review_length_distribution.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved review length distribution plot to '{output_path}'.")
    return stats


def plot_wordcloud(
    df: pd.DataFrame,
    sentiment: str,
    text_col: str = "cleaned_review",
    sentiment_col: str = "Sentiment",
    output_dir: str = "outputs/figures",
) -> plt.Figure:
    """
    Generate and save a word cloud visualization for a specific sentiment class with explicit legend badge.

    Args:
        df (pd.DataFrame): Full labeled DataFrame.
        sentiment (str): Sentiment class ('Positive', 'Neutral', or 'Negative').
        text_col (str): Column name containing text.
        sentiment_col (str): Column name containing sentiment labels.
        output_dir (str): Output directory.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    os.makedirs(output_dir, exist_ok=True)
    subset_text = " ".join(df[df[sentiment_col] == sentiment][text_col].dropna())

    if not subset_text.strip():
        raise ValueError(f"No text available for sentiment class '{sentiment}'.")

    cmap_dict = {"Positive": "Greens", "Neutral": "Blues", "Negative": "Reds"}

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=cmap_dict.get(sentiment, "viridis"),
        max_words=150,
        random_state=42,
    ).generate(subset_text)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud Visualization - {sentiment} Sentiment Reviews", fontsize=14, fontweight="bold", pad=12)

    # Explicit legend patch indicating class color theme
    class_patch = mpatches.Patch(color=PALETTE.get(sentiment, "#333"), label=f"Class Theme: {sentiment} Words")
    ax.legend(handles=[class_patch], loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"wordcloud_{sentiment.lower()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {sentiment} wordcloud to '{output_path}'.")
    return fig


def plot_top_ngrams(
    df: pd.DataFrame,
    sentiment: str,
    n: int = 20,
    ngram_range: tuple = (2, 2),
    text_col: str = "cleaned_review",
    sentiment_col: str = "Sentiment",
    output_dir: str = "outputs/figures",
) -> plt.Figure:
    """
    Plot and save the top-N n-grams for a specific sentiment class with legend and detailed axis labels.

    Args:
        df (pd.DataFrame): Full labeled DataFrame.
        sentiment (str): Sentiment class.
        n (int): Number of top n-grams to display (default 20).
        ngram_range (tuple): Tuple specifying (min_n, max_n) (default (2, 2) for bigrams).
        text_col (str): Column name containing text.
        sentiment_col (str): Column name containing sentiment.
        output_dir (str): Output directory.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    os.makedirs(output_dir, exist_ok=True)
    subset = df[df[sentiment_col] == sentiment][text_col].dropna()

    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=1000)
    bag_of_words = vec.fit_transform(subset)
    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

    ngram_df = pd.DataFrame(words_freq, columns=["ngram", "count"])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bar_color = PALETTE.get(sentiment, "#34495e")
    bars = ax.barh(ngram_df["ngram"][::-1], ngram_df["count"][::-1], color=bar_color, edgecolor="black", linewidth=0.5)

    # Value annotations on bar tips
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width):,}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_title(f"Top {n} Most Frequent Bigrams - {sentiment} Sentiment", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Frequency Count (Occurrences in Reviews)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Bigram (2-Word Sequence)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlim(0, max(ngram_df["count"]) * 1.15)

    # Legend
    legend_patch = mpatches.Patch(color=bar_color, label=f"{sentiment} Bigram Frequencies")
    ax.legend(handles=[legend_patch], loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"top_{n}_bigrams_{sentiment.lower()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved top {n} bigrams for {sentiment} to '{output_path}'.")
    return fig
