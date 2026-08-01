# Google Play Analytics: SQL, Data Analytics & NLP Sentiment Classification

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-red.svg)](https://pytorch.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-orange.svg)](https://www.mysql.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-ff4b4b.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data analytics and natural language processing (NLP) project built on the Google Play Store dataset. This repository demonstrates relational data modeling in MySQL, analytical SQL querying, classical machine learning baseline models (Logistic Regression, Linear SVM), fine-tuning a transformer model (DistilBERT), and deploying an interactive multi-page Streamlit web application.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [SQL Pipeline](#sql-pipeline)
- [Business Insights](#business-insights)
- [NLP Pipeline](#nlp-pipeline)
- [Model Comparison](#model-comparison)
- [Streamlit Application](#streamlit-application)
- [Dashboard Preview](#dashboard-preview)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Technology Stack](#technology-stack)
- [Author](#author)

---

## Project Overview

### Business Problem
The Google Play Store contains millions of mobile applications competing for user acquisition and retention. Product managers and app developers face two primary challenges:
1. **Understanding Market Dynamics**: Identifying category competition, pricing model trade-offs (Free vs. Paid), and install reach patterns across different app verticals.
2. **Analyzing Customer Feedback at Scale**: Review volume makes manual inspection impractical. Unstructured user feedback contains critical signals regarding app crashes, feature requests, and user dissatisfaction.

### Project Objectives
- Build a relational MySQL Star Schema database to store clean app metadata and user review records.
- Implement 37 analytical SQL queries to extract market performance, rating distributions, pricing ceilings, and category health metrics.
- Train and benchmark classical ML baseline models against a fine-tuned DistilBERT transformer for 3-class sentiment classification (Positive, Neutral, Negative).
- Deploy an interactive Streamlit application enabling business analytics exploration, single-review real-time sentiment prediction, and bulk CSV batch prediction.

### Real-World Motivation
App ratings directly influence store search algorithms and conversion rates. Automatically identifying negative sentiment spikes helps product teams prioritize bug fixes before customer churn affects public ratings.

---

## Key Features

### SQL & Data Processing
- **Data Sanitization**: Handling missing ratings, parsing currency values, and converting install strings into numeric formats.
- **Relational Schema**: MySQL Star Schema design separating application dimensions (`dim_apps`) from review facts (`fact_user_reviews`).
- **Data Validation Script**: Automated SQL script (`sql/validate_etl.sql`) checking referential integrity, duplicate records, and null values.
- **37 Analytical SQL Queries**: Queries covering dense rankings, rating tiers, price boxplots, and review polarity.

### Business Analytics
- **Multi-Tab Dashboard**: Interactive Streamlit interface organized into *App Performance*, *Customer Sentiment*, *Revenue & Pricing*, *Category Insights*, and *Market Trends*.
- **Business Insights**: Structured callouts under every visualization highlighting Key Insight, Business Interpretation, and Suggested Business Action.
- **Interactive Filtering**: Real-time filtering by App Category, Monetisation Model (Free/Paid), and Customer Sentiment.

### Machine Learning & NLP
- **Text Preprocessing**: Lowercasing, punctuation stripping, stop-word filtering, and NLTK tokenization.
- **TF-IDF Feature Extraction**: Unigram and bigram representation (20,000 max features) for classical baselines.
- **Model Training**: 5-fold cross-validated `GridSearchCV` for Logistic Regression and Linear SVM.
- **DistilBERT Fine-Tuning**: PyTorch fine-tuning of `distilbert-base-uncased` with class-weighted loss and Automatic Mixed Precision (AMP).
- **Inference Engines**: Real-time single review prediction and bulk CSV batch processing.

---

## Dataset

This project utilizes two complementary Google Play Store datasets originally sourced from Kaggle:

1. **Google Play Store Apps (`googleplaystore.csv`)**:
   - Contains 10,841 application entries with attributes including `App`, `Category`, `Rating`, `Reviews`, `Size`, `Installs`, `Type`, `Price`, `Content Rating`, `Genres`, `Last Updated`, `Current Ver`, and `Android Ver`.

2. **Google Play User Reviews (`googleplaystore_user_reviews.csv`)**:
   - Contains 64,295 review entries with attributes including `App`, `Translated_Review`, `Sentiment`, `Sentiment_Polarity`, and `Sentiment_Subjectivity`.

### Data Privacy & Storage Notice
Raw dataset files are excluded from git tracking via `.gitignore`. 

### Download & Setup Instructions
To run the dataset pipelines locally:
1. Download the dataset files from Kaggle: [Google Play Store Apps Dataset](https://www.kaggle.com/datasets/lava18/google-play-store-apps).
2. Place `googleplaystore.csv` and `googleplaystore_user_reviews.csv` into the `data/` directory:
   ```text
   Google_Play_Analytics/
   └── data/
       ├── googleplaystore.csv
       └── googleplaystore_user_reviews.csv
   ```

---

## Project Workflow

```text
Google Play Store Datasets (Kaggle)
               │
               ▼
┌─────────────────────────────┐
│  Data Cleaning & ETL        │  (src/preprocessing.py)
└──────────────┬──────────────┘
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│  MySQL Star Schema Database │           │  NLP Corpus Preprocessing   │
│  (dim_apps, fact_reviews)   │           │  (70% Train, 15% Val, 15%) │
└──────────────┬──────────────┘           └──────────────┬──────────────┘
               │                                         │
               ▼                                         ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│  37 Analytical SQL Queries  │           │  Model Training Pipeline    │
│  (sql/analytics_queries.sql)│           │  (LR, SVM, DistilBERT GPU)  │
└──────────────┬──────────────┘           └──────────────┬──────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Streamlit Portal            │
                     │ (app.py & views/)           │
                     └─────────────────────────────┘
```

---

## Repository Structure

```text
Google_Play_Analytics/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusion rules
├── data/                       # Local dataset directory (not tracked in git)
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── sql/                        # SQL scripts directory
│   ├── etl_star_schema.sql     # Database table DDL schema creation
│   ├── validate_etl.sql        # Data validation and integrity checks
│   └── analytics_queries.sql   # 37 analytical SQL business queries
├── src/                        # Core Python package modules
│   ├── utils.py                # Helper utilities (seed, device, serialization)
│   ├── preprocessing.py        # Data loading, cleaning, and dataset splitting
│   ├── sentiment.py            # Model training (LR, SVM, DistilBERT)
│   └── classifier.py           # Evaluation pipeline and metric calculations
├── views/                      # Streamlit page view components
│   ├── home.py                 # Executive Home page
│   ├── sql_dashboard.py        # Business Insights Dashboard page
│   ├── single_prediction.py    # Real-time Predict Review page
│   ├── batch_prediction.py     # Batch Analysis CSV upload page
│   └── about.py                # About Project & Technical Details page
├── outputs/                    # Exported project artifacts
│   ├── cleaned_data/           # Processed CSV data splits (train, val, test)
│   ├── models/                 # Model checkpoints (joblib, safetensors, json)
│   ├── predictions/            # Test set predictions and classification reports
│   └── figures/                # Confusion matrix heatmap plots (PNG)
└── scratch/                    # Verification scripts
```

---

## SQL Pipeline

The SQL pipeline establishes an optimized relational star schema in MySQL.

### Database Schema
- **`dim_apps` (Dimension Table)**:
  - Primary Key: `app_id`
  - Attributes: `app_name`, `category`, `rating`, `reviews_count`, `size_in_mb`, `installs_count`, `is_paid`, `price_usd`, `content_rating`, `genres`, `last_updated_date`, `min_android_ver`.
- **`fact_user_reviews` (Fact Table)**:
  - Primary Key: `review_id`
  - Foreign Key: `app_id` (references `dim_apps.app_id`)
  - Attributes: `translated_review`, `sentiment`, `sentiment_polarity`, `sentiment_subjectivity`.

### Data Validation (`sql/validate_etl.sql`)
Checks performed before executing analytics:
- Verification of row counts and column data types.
- Check for orphan review records lacking a matching `app_id`.
- Identification of duplicate application entries and out-of-range rating values.

### Analytical SQL Queries (`sql/analytics_queries.sql`)
The repository contains 37 SQL queries divided into 9 analytical groups:
1. **App Performance**: Top install leaderboards and dense quality rankings.
2. **Rating Analytics**: Rating distributions and outlier identification.
3. **Category Analysis**: Category app counts, average sizes, and install volumes.
4. **Pricing Analysis**: Free vs. paid rating comparisons and category price ceilings.
5. **Install Analysis**: Install bucket distributions and median install volume.
6. **Review Analysis**: High-review app rankings and text completeness checks.
7. **Sentiment Analysis**: Store-wide sentiment splits and category polarity averages.
8. **Business Intelligence**: Composite quality-reach scores and pricing polarity gaps.
9. **Executive KPIs**: Store snapshot metrics.

---

## Business Insights

The dashboard addresses strategic questions relevant to app developers and product managers:

- **App Category Performance**: Which categories generate the highest cumulative install volume? *(Gaming, Communication, and Tools lead overall installs).*
- **Rating Distributions**: How are ratings distributed across the store? *(Over 75% of rated apps score above 4.0 stars, establishing 4.0 as the baseline threshold for user trust).*
- **Monetisation Trade-Offs**: Do paid apps receive better ratings than free apps? *(Paid apps average 4.26 stars compared to 4.18 stars for free apps, reflecting ad-free user experiences).*
- **Customer Feedback Patterns**: What is the overall sentiment split across user reviews? *(64.7% Positive, 23.9% Negative, 11.4% Neutral).*
- **Quality-Reach Opportunities**: Which app verticals offer high install reach relative to competitor saturation?

---

## NLP Pipeline

The NLP pipeline prepares review text and fine-tunes a transformer model for sentiment classification.

```text
Raw Review Text ──► Text Cleaning ──► Tokenization ──► Sequence Truncation (max_len=64) ──► DistilBERT ──► Logits ──► Softmax
```

### Preprocessing & Dataset Splitting
- **Corpus**: 27,992 valid labeled reviews after dropping missing values and duplicate text entries.
- **Labels**: Mapped to numerical IDs: `Negative: 0`, `Neutral: 1`, `Positive: 2`.
- **Split Ratio**: Stratified split into 70% Training (`19,594` samples), 15% Validation (`4,199` samples), and 15% Test (`4,199` samples).

### Feature Engineering & Modeling
- **TF-IDF Vectorization**: Configured with unigram/bigram ranges (`1, 2`) and 20,000 max features for baseline models.
- **DistilBERT Fine-Tuning**:
  - Model: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions).
  - Sequence Length: `max_length = 64` tokens (covers **97.14%** of reviews without truncation).
  - Loss Function: Weighted `CrossEntropyLoss` ($w = [1.50, 2.25, 0.53]$) to address class imbalance.
  - Hardware Acceleration: Fine-tuned on an NVIDIA GeForce RTX 2050 GPU using Automatic Mixed Precision (`BF16`/`FP16`).

### Model Hosting & Deployment
The fine-tuned DistilBERT model (approx. 267MB) exceeds GitHub's standard 100MB file size limit and is therefore excluded from this Git repository. 
For deployment, the model is hosted securely on the Hugging Face Model Hub:
- [Hugging Face Repository: Samayita-23/google-play-distilbert-sentiment](https://huggingface.co/Samayita-23/google-play-distilbert-sentiment)

The Streamlit application is configured to prioritize any local model files. If the local model is not found (e.g., in a cloud deployment environment), it automatically downloads and loads the fine-tuned model directly from Hugging Face at runtime, ensuring a seamless deployment process without requiring Git LFS.

---

## Model Comparison

Three models were evaluated on the exact same held-out test dataset (`4,199` samples).

### Evaluated Models
1. **Logistic Regression**: Trained on TF-IDF features with balanced class weights and 5-fold cross-validation.
2. **Linear SVM (`LinearSVC`)**: Trained on TF-IDF features with calibrated decision scores for ROC-AUC.
3. **DistilBERT**: Fine-tuned sequence classification transformer.

### Test Set Performance Benchmarks

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 (Primary) | Weighted F1 | ROC-AUC (OvR) | Training Time | Inference Time (4,199 samples) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 85.90% | 0.8120 | 0.8443 | 0.8266 | 0.8610 | 0.9520 | 0m 16s | 0.0050s |
| **Linear SVM** | 86.09% | 0.8186 | 0.8340 | 0.8258 | 0.8618 | 0.9530 | 0m 04s | 0.0021s |
| 🌟 **DistilBERT** | **92.12%** | **0.9029** | **0.9122** | **0.9066** | **0.9221** | **0.9819** | 5m 19s | 12.5365s |

### Model Selection Rationale
**DistilBERT was selected for deployment** because it achieved the highest **Test Macro F1 score of 0.9066** (an **+8.0% absolute improvement** over classical baselines). DistilBERT's bidirectional self-attention mechanism effectively captures context, word order, and sentiment negations (*"not bad at all"* vs. *"bad update"*) that bag-of-words TF-IDF models misclassify.

---

## Streamlit Application

The Streamlit web application is structured into five user-focused pages:

1. **🏠 Home**: Executive overview, top-line KPI cards, business learning points, and portal module navigation.
2. **📊 Business Insights**: Visualizations categorized into *App Performance*, *Customer Sentiment*, *Revenue & Pricing*, *Category Insights*, and *Market Trends*. Includes interactive filtering and an analytical query selector.
3. **🤖 Predict Review**: Real-time sentiment prediction for custom review text inputs with confidence scores, probability breakdown charts, and actionable guidance for product teams.
4. **📁 Batch Analysis**: CSV file upload interface for batch sentiment prediction, dataset sentiment summaries, health status checks, and downloadable CSV output reports.
5. **ℹ️ About Project**: Non-technical project overview paired with an expandable **Technical Details** section containing model benchmark tables, confusion matrix heatmaps, classification reports, and architecture specs.

---

## Dashboard Preview

```text
+-----------------------------------------------------------------------------------+
|  [Screenshot Placeholder: Home Page Executive Overview & KPI Snapshot]            |
+-----------------------------------------------------------------------------------+
|  [Screenshot Placeholder: Business Insights Category & Rating Visualizations]     |
+-----------------------------------------------------------------------------------+
|  [Screenshot Placeholder: Predict Review Real-Time Sentiment Interface]            |
+-----------------------------------------------------------------------------------+
|  [Screenshot Placeholder: Batch Analysis CSV Dataset Overview & Download]          |
+-----------------------------------------------------------------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- MySQL Server 8.0+ (Optional for local database execution)
- NVIDIA GPU with CUDA drivers (Optional for GPU acceleration; CPU fallback supported)

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SquirrelCat235/Google-Play-Store-Analytics.git
   cd Google-Play-Store-Analytics
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables Configuration (Optional)**:
   Copy `.env.example` to `.env` and set your local MySQL connection settings:
   ```bash
   cp .env.example .env
   ```
   The `.env` configuration template supports the following variables:
   ```env
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=
   DB_NAME=Google_Play
   ```

---

## Running the Project

### 1. Database Setup (Optional)
Ensure MySQL is running, then configure your database connection parameters via environment variables in `.env` or via CLI arguments.
Execute the SQL validation runner:
```bash
python run_sql_tests.py --user YOUR_MYSQL_USER --password YOUR_MYSQL_PASSWORD
```
Or import raw CSV data into staging tables:
```bash
python import_csv.py --user YOUR_MYSQL_USER --password YOUR_MYSQL_PASSWORD
```

### 2. Run Training & Evaluation Pipelines (Pre-Trained Models Included)
If you wish to re-execute the ML or evaluation pipelines manually:
```bash
# Classical ML Pipeline
python src/sentiment.py

# Model Evaluation Pipeline
python src/classifier.py
```

### 3. Launch Streamlit Web Application
Run the Streamlit app locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Results

- **Data Engineering**: Established a clean MySQL relational database schema and validated integrity across 10,841 apps and 37,427 user reviews.
- **Analytics**: Derived actionable insights across 37 SQL business queries, identifying pricing trends and category competition dynamics.
- **Sentiment Classification**: Fine-tuned DistilBERT to achieve **92.12% accuracy** and **0.9066 Macro F1** on 4,199 test reviews.
- **Deployment**: Delivered an interactive Streamlit portal enabling real-time review sentiment inference and batch CSV processing.

---

## Limitations

- **Sarcasm and Subtle Context**: Like most transformer models, DistilBERT can occasionally misclassify sarcasm or complex double-negatives.
- **Language Scope**: The current dataset and model are trained exclusively on English-language user reviews.
- **Sequence Truncation**: Sequence length is capped at `max_length = 64` tokens to optimize memory usage (covering 97.14% of reviews in full, but truncating longer text entries).

---

## Future Improvements

- **Multi-Lingual Sentiment Analysis**: Fine-tune XLM-RoBERTa to process non-English global user reviews.
- **Explainable AI (XAI)**: Integrate SHAP or LIME to visualize token-level attribution weights for individual sentiment predictions.
- **Topic Modeling**: Implement BERTopic or LDA to group negative reviews into specific feature bug clusters.
- **Automated Data Ingestion**: Set up a web scraping pipeline to automatically ingest live Google Play Store reviews daily.
- **Containerization & Deployment**: Package the application into a Docker container for deployment to AWS ECS or GCP Cloud Run.

---

## Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Programming Languages** | Python 3.13, SQL |
| **Database** | MySQL 8.0, SQLAlchemy, PyMySQL |
| **Machine Learning** | Scikit-Learn, Joblib |
| **NLP & Deep Learning** | PyTorch 2.6, HuggingFace Transformers (`distilbert-base-uncased`), NLTK |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Plotly Express, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Development Tools** | Git, VS Code |

---

## Author

- **Name**: Samayita Mohanta
- **GitHub**: [SquirrelCat235](https://github.com/SquirrelCat235)
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/samayita-mohanta-1a549b325/)
