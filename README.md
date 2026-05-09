# Multi-Label Text Classification of AI Behavioral Impact: An NLP & Quantitative Finance Pipeline
### 🥈 2nd Place · RAISE-26 AI-NLP Informatics Competition · Rutgers University

---

## 📌 Abstract

This project investigates how AI-related news media frames human behavioral impact through a multi-label text classification framework applied to 10,500 news headlines across a custom 12-category behavioral taxonomy. We benchmark a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT transformer, conduct cross-LLM distributional analysis across Mistral, Qwen, and Llama, and extend findings into a quantitative finance layer examining predictive relationships between NLP-derived behavioral signals and equity market volatility.

**Research areas:** Multi-Label Text Classification · Transformer Fine-Tuning · Feature Engineering · Statistical Hypothesis Testing · Time Series Econometrics · Cross-LLM Behavioral Analysis

---

## 🔬 Methodology

### Stage 1–4: Data Pipeline & Preprocessing
- Corpus: 10,500 AI-related news headlines (Dataset A), annotated with a custom 12-label behavioral impact taxonomy
- Text preprocessing: lowercasing, URL/HTML removal, whitespace normalization
- Multi-hot label encoding with `MultilabelStratifiedShuffleSplit` (iterative stratification) to preserve label co-occurrence distributions across 70/10/20 train/val/test splits
- Label frequency analysis, co-occurrence mapping, and distribution validation across splits

### Stage 5: Baseline — TF-IDF + Logistic Regression
- Feature engineering: TF-IDF vectorization with unigram and bigram features (`ngram_range=(1,2)`, `max_features=60,000`, `min_df=2`)
- Classifier: `OneVsRestClassifier` wrapping `LogisticRegression` (`class_weight=balanced`, `solver=liblinear`, `max_iter=2000`)
- Decision threshold: 0.50 on sigmoid-calibrated probabilities
- **Macro-F1: 0.9331 · Micro-F1: 0.9430**

### Stage 6: Deep Learning — DistilBERT Fine-Tuning
- Base model: `distilbert-base-uncased` (66M parameters, 6-layer transformer)
- Architecture: CLS token representation → Dropout (p=0.3) → Linear classification head (12 outputs)
- Training: AdamW optimizer · BCEWithLogitsLoss · 3 epochs · batch size 16 · lr=2e-5
- Custom `MultiLabelDataset` class with PyTorch `DataLoader` and attention mask handling
- **Macro-F1: 0.8862**

### Stage 7–8: Evaluation & Interpretability
- Per-label precision, recall, F1, and support analysis across all 12 behavioral categories
- Confusion matrix analysis (TP/FP/FN/TN) for top labels
- TF-IDF coefficient extraction: top positive and negative feature weights per label
- High-confidence error analysis: false positives and false negatives by probability threshold

### Stage 9: Unsupervised Topic Modeling
- Non-negative Matrix Factorization (NMF) with `n_components=10`
- TF-IDF vectorization (`max_features=5,000`, `ngram_range=(1,2)`, `min_df=3`)
- Latent topic extraction and comparison against supervised behavioral label taxonomy

### Stage 10: Cross-LLM Behavioral Analysis
- Dataset C: LLM-generated text from Mistral, Qwen, and Llama (competition-provided; generation parameters not controlled)
- Applied trained classifier to generate per-LLM label distributions
- **Statistical testing:**
  - Chi-square test of independence: χ²=74.21, p=1.41×10⁻⁷
  - Cramér's V = 0.065 (statistically significant; small practical effect size)
  - Jensen-Shannon divergence for pairwise distributional similarity
- **Finding:** All three LLMs exhibit highly convergent behavioral framing of AI's societal impact despite architectural differences, suggesting emergent consensus in how large language models represent AI-human behavioral dynamics

### Stage 11: Quantitative Financial Analysis
- Equities: NVDA, GOOGL, MSFT, META, QQQ, SPY
- **GARCH(1,1)** conditional volatility modeling with annualized volatility estimation
- **ADF stationarity testing** on return time series prior to causality analysis
- **Granger causality testing** (lag selection via AIC) — NLP behavioral signals as predictors of market volatility
- **Pearson and Spearman correlation** between daily label proportions and equity returns/volatility
- Industry exposure matrix linking behavioral label categories to sector dynamics

---

## 📊 Key Results

| Model | Macro-F1 | Micro-F1 | Training Time |
|---|---|---|---|
| TF-IDF + Logistic Regression | **0.9331** | **0.9430** | ~5 min (CPU) |
| DistilBERT (`distilbert-base-uncased`) | 0.8862 | ~0.9400 | 30–60 min (GPU) |

> **Counterintuitive finding:** The classical baseline outperformed the fine-tuned transformer by ~4.69 Macro-F1 points. At avg. 10–15 tokens per headline, input sequences are too short for DistilBERT's self-attention mechanism to capture meaningful contextual dependencies — sparse TF-IDF features with discriminative n-gram vocabulary proved more effective for this domain at 6–12x lower computational cost.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Environment | Google Colab |
| Deep Learning | PyTorch · Transformers (`distilbert-base-uncased`) |
| ML / Classification | scikit-learn · iterative-stratification |
| NLP | NLTK · TF-IDF · NMF · n-gram extraction |
| Feature Engineering | Multi-hot encoding · bigram vectorization · TF-IDF coefficient analysis |
| Statistical Modeling | scipy · statsmodels |
| Time Series & Econometrics | arch (GARCH) · Granger causality · ADF test |
| Data Processing | pandas · NumPy |
| Visualization | matplotlib · seaborn |

---

## 🚀 Getting Started

1. Open `RAISE26_Synaptic_Sparks_copy.ipynb` in **Google Colab**
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
