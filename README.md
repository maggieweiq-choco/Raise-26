# Multi-Label NLP Pipeline & Quantitative Signal Analysis
### 🥈 2nd Place · RAISE-26 AI-NLP Informatics Competition · Rutgers University

---

## 📌 Overview

An end-to-end machine learning pipeline that classifies 10,500 AI-related news headlines across a custom 12-category behavioral taxonomy, compares classical and transformer-based NLP models, analyzes cross-LLM behavioral patterns, and links NLP-derived signals to equity market dynamics.

---

## 🔬 Pipeline (11 Stages)

| Stage | Description |
|---|---|
| 1 | Environment Setup & Reproducibility |
| 2 | Data Loading & Exploratory Analysis |
| 3 | Text Preprocessing & Label Handling |
| 4 | Multi-Hot Encoding & Iterative Stratified Splitting |
| 5 | Baseline Model: TF-IDF + Logistic Regression |
| 6 | Deep Learning: DistilBERT Fine-Tuning |
| 7 | Model Evaluation & Comparison |
| 8 | Interpretability Analysis |
| 9 | Topic Modeling (NMF) |
| 10 | Cross-LLM Behavioral Analysis |
| 11 | Quantitative Financial Analysis |

---

## 🤖 Models & Methods

### TF-IDF + Logistic Regression (Baseline)
- TF-IDF vectorization with bigram features (`ngram_range=(1,2)`, `max_features=60,000`)
- OneVsRest Logistic Regression with `class_weight=balanced`, `solver=liblinear`
- Multi-hot label encoding with iterative stratification for balanced splits
- **Macro-F1: 0.9331 · Micro-F1: 0.9430**

### DistilBERT (Fine-Tuned)
- Model: `distilbert-base-uncased`
- Optimizer: AdamW · Loss: BCEWithLogitsLoss · Epochs: 3
- Custom `MultiLabelDataset` class with PyTorch DataLoader
- Dropout regularization (p=0.3) on CLS representation
- **Macro-F1: 0.8862**

### Key Finding
> TF-IDF outperformed DistilBERT by ~4.69 Macro-F1 points. Short headline length (avg. 10–15 tokens) limits transformer contextual advantage — classical sparse features captured discriminative vocabulary more effectively at this scale.

---

## 🔀 Cross-LLM Behavioral Analysis

Compared label distributions across outputs from **Mistral, Qwen, and Llama** using the trained classifier.

- Chi-square test: χ²=74.21, p=1.41×10⁻⁷
- Cramér's V = 0.065 — statistically significant, small practical effect
- Jensen-Shannon divergence for pairwise similarity measurement
- **Finding:** All three LLMs converge on similar behavioral framing of AI's societal impact despite architectural differences

---

## 📈 Quantitative Financial Analysis

Linked NLP-derived behavioral signals to equity market dynamics across **NVDA, GOOGL, MSFT, META**.

- **GARCH(1,1)** volatility modeling — annualized conditional volatility estimates
- **Granger causality testing** — whether news behavioral signals predict market volatility
- **Pearson/Spearman correlation** — news label proportions vs. market returns
- **ADF stationarity testing** for time series preprocessing

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch · Transformers |
| ML / NLP | scikit-learn · NLTK · iterative-stratification |
| Statistical Analysis | scipy · statsmodels · arch |
| Data | pandas · NumPy |
| Visualization | matplotlib · seaborn |
| Environment | Google Colab |

---

## 🚀 Getting Started

1. Open `RAISE26_Synaptic_Sparks_copy.ipynb` in Google Colab
2. Install dependencies:
```python
pip install pandas numpy matplotlib seaborn nltk scikit-learn transformers torch accelerate iterative-stratification yfinance arch
```
3. Upload dataset files when prompted
4. Run all cells sequentially

> **Note:** Datasets are not included as they are competition-provided and not for public distribution.

---

## 👥 Team · Synaptic Sparks

| Name | GitHub |
|---|---|
| Ziqi Wei (Maggie) | [@maggieweiq-choco](https://github.com/maggieweiq-choco) |
| Keqi Zhang | [@KiraZhang-Keqi](https://github.com/KiraZhang-Keqi) |
| Junemo Moon | [@Junemo-hub](https://github.com/Junemo-hub) |
