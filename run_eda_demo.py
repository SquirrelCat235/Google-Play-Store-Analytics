"""
Run EDA Demo Viewer
-------------------
Launches the interactive HTML demo window in your default web browser
and pops up the Matplotlib EDA visualization figure window.
"""

import os
import sys
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    html_demo = os.path.join(project_dir, "demo_window.html")
    figures_dir = os.path.join(project_dir, "outputs", "figures")

    print("==================================================")
    print(" Google Play Analytics - EDA Diagrams Demo Viewer ")
    print("==================================================")

    # 1. Open Interactive HTML Demo Window in default web browser
    if os.path.exists(html_demo):
        print("--> Opening Interactive HTML Demo Window in web browser...")
        webbrowser.open(f"file:///{html_demo.replace(os.sep, '/')}")
    else:
        print(f"--> [Warning] {html_demo} not found.")

    # 2. Display figures in Matplotlib GUI Grid Window
    fig_files = [
        "sentiment_distribution.png",
        "review_length_distribution.png",
        "wordcloud_positive.png",
        "wordcloud_negative.png",
        "wordcloud_neutral.png",
        "top_20_bigrams_positive.png",
        "top_20_bigrams_negative.png",
        "top_20_bigrams_neutral.png",
    ]

    valid_figs = [f for f in fig_files if os.path.exists(os.path.join(figures_dir, f))]

    if not valid_figs:
        print(f"--> No figure PNG files found in '{figures_dir}'.")
        return

    print(f"--> Displaying {len(valid_figs)} EDA diagrams in Matplotlib GUI window...")

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("Google Play Analytics - Exploratory Data Analysis (EDA) Diagrams", fontsize=16, fontweight="bold", y=0.995)

    axes_flat = axes.flatten()
    for idx, fname in enumerate(valid_figs):
        fpath = os.path.join(figures_dir, fname)
        img = mpimg.imread(fpath)
        axes_flat[idx].imshow(img)
        axes_flat[idx].set_title(fname.replace(".png", "").replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes_flat[idx].axis("off")

    plt.tight_layout()
    print("--> Plot window open. Close plot window when finished.")
    plt.show()


if __name__ == "__main__":
    main()
