# 🥈 RAISE-26 · Synaptic Sparks
### 2nd Place · RAISE-26 AI-NLP Informatics Competition · Rutgers University (Bloustein School, MPI Program)

> **"Mirror, Mirror on the Wall, Is AI Transforming Us All?"**  
> An end-to-end NLP classification pipeline and quantitative signal analysis investigating how AI news frames human behavioral impact.

---

## 🏆 Competition
| | |
|---|---|
| **Competition** | RAISE-26 AI-NLP Informatics Competition |
| **Host** | Rutgers University, Bloustein School MPI Program |
| **Result** | 🥈 2nd Place (17 teams, including 7 graduate-level teams) |
| **Team** | Synaptic Sparks — Ziqi Wei · Junemo Moon · Keqi Zhang |

---

## 📌 Project Overview

This project builds a multi-label text classification system on 10,500 AI-related news headlines annotated across a custom 12-category behavioral taxonomy. We compare a classical NLP baseline against a fine-tuned transformer model, extend the analysis to cross-LLM behavioral comparison, and integrate a quantitative finance layer linking news signals to equity market dynamics.

---

## 🔬 Pipeline Overview (11 Stages)

| Stage | Description |
|---|---|
| 1 | Environment Setup & Reproducibility |
| 2 | Data Loading & Exploratory Analysis |
| 3 | Text Preprocessing & Label Handling |
| 4 | Multi-Hot Encoding & Iterative Stratified Splitting |
| 5 | Baseline Model: TF-IDF + Logistic Regression |
| 6 | Deep Learning Model: DistilBERT Fine-Tuning |
| 7 | Model Evaluation & Comparison |
| 8 | Interpretability Analysis |
| 9 | Topic Modeling (NMF) |
| 10 | Cross-LLM Behavioral Analysis (Dataset C) |
| 11 | Quantitative Financial Analysis |

---

## 📊 Key Results

### Model Comparison
| Model | Macro-F1 | Micro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | **0.9331** | 0.9430 |
| DistilBERT (`distilbert-base-uncased`) | 0.8862 | ~0.9400 |

> The classical baseline outperformed DistilBERT by ~4.69 Macro-F1 points. News headlines average only 10–15 tokens, limiting the contextual advantage of transformer-based models. DistilBERT was fine-tuned with AdamW optimizer, BCEWithLogitsLoss, over 3 epochs.

### Cross-LLM Behavioral Analysis (Mistral · Qwen · Llama)
- Dataset C was competition-provided; generation parameters were not controlled by the team
- Chi-square test: χ²=74.21, p=1.41×10⁻⁷
- Cramér's V = 0.065 — statistically significant, but small practical effect size
- **Finding:** All three LLMs show highly convergent behavioral framing of AI's societal impact, with only minor distributional differences across the 12 behavioral categories

### Quantitative Finance (Stage 11)
- GARCH(1,1) volatility modeling across NVDA, GOOGL, MSFT, META
- Granger causality testing between NLP-derived behavioral signals and equity returns
- Pearson/Spearman correlation analysis between news label proportions and market volatility

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch · Transformers (`distilbert-base-uncased`) |
| ML / NLP | scikit-learn · NLTK · iterative-stratification |
| Statistical Analysis | scipy · statsmodels · arch |
| Data | pandas · NumPy |
| Visualization | matplotlib · seaborn |
| Environment | Google Colab |

---

## 📁 Repository Structure

```
RAISE26-Synaptic-Sparks/
│
├── RAISE26_Synaptic_Sparks_copy.ipynb   # Main pipeline (final submission)
├── README.md
│
└── docs/
    ├── RAISE-26_Competition_Guidelines.docx
    └── RAISE-26_Helpful_Resources.docx
```

> **Note:** Datasets are not included in this repository as they are competition-provided and not for public distribution.

---

## 🚀 Getting Started

This notebook is designed to run on **Google Colab**.

1. Open `RAISE26_Synaptic_Sparks_copy.ipynb` in Google Colab
2. Run the environment setup cell to install all dependencies:
```python
pip install pandas numpy matplotlib seaborn nltk scikit-learn transformers torch accelerate iterative-stratification yfinance arch
```
3. Upload the required dataset files when prompted
4. Run all cells sequentially

---

## 👥 Team

| Name | GitHub |
|---|---|
| Ziqi Wei (Maggie) | [@maggieweiq-choco](https://github.com/maggieweiq-choco) |
| Keqi Zhang | [@KiraZhang-Keqi](https://github.com/KiraZhang-Keqi) |
| Junemo Moon | [@Junemo-hub](https://github.com/Junemo-hub) |

---

## 🔗 Links
- [RAISE-26 Competition Page](https://raise26.devpost.com/)
- [Rutgers MPI Program](https://bloustein.rutgers.edu/graduate/public-informatics/)
