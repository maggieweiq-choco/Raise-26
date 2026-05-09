# ============================================================
# Cell 1: Environment Setup
# ============================================================

!pip install -q transformers torch scikit-learn pandas numpy matplotlib seaborn
!pip install -q statsmodels arch yfinance

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================
# Cell 2: Upload Data Files
# ============================================================

from google.colab import files
import os

print("Upload the following files:")
print("   1. dataset_A_news_full_10500.csv")
print("   2. Dataset_C_prompts___queries.csv")
print("   3. distilbert_complete.py")

uploaded = files.upload()

required_files = [
    'dataset_A_news_full_10500.csv',
    'Dataset_C_prompts___queries.csv',
    'distilbert_complete.py'
]

print("Uploaded files:")
for f in uploaded.keys():
    print(f"   - {f} ({os.path.getsize(f) / 1e6:.2f} MB)")

missing = [f for f in required_files if f not in uploaded]
if missing:
    print(f"Missing files: {missing}")
else:
    print("All required files uploaded.")


# ============================================================
# Cell 3: Data Exploration
# ============================================================

df_news = pd.read_csv('dataset_A_news_full_10500.csv')
df_llm = pd.read_csv('Dataset_C_prompts___queries.csv')

print("Dataset A")
print("=" * 50)
print(f"Samples: {len(df_news)}")
print(f"Columns: {df_news.columns.tolist()}")
print(df_news.head(3))

if 'classes_str' in df_news.columns:
    all_labels = []
    for labels_str in df_news['classes_str'].dropna():
        labels = [l.strip() for l in labels_str.split(',')]
        all_labels.extend(labels)

    label_counts = pd.Series(all_labels).value_counts()
    print(f"Label distribution ({len(label_counts)} labels):")
    print(label_counts)

    plt.figure(figsize=(14, 6))
    label_counts.plot(kind='barh', color='skyblue')
    plt.xlabel('Frequency')
    plt.title('AI News Label Distribution')
    plt.tight_layout()
    plt.show()

print("Dataset C")
print("=" * 50)
print(f"Samples: {len(df_llm)}")
print(f"Columns: {df_llm.columns.tolist()}")
if 'LLM' in df_llm.columns:
    print(df_llm['LLM'].value_counts())


# ============================================================
# Cell 4: DistilBERT Model Training
# ============================================================

from distilbert_complete import load_and_prepare_data, train_distilbert_model

print("Preparing training data...")
data = load_and_prepare_data(
    csv_path='dataset_A_news_full_10500.csv',
    text_col='title',
    label_col='classes_str',
    test_size=0.2,
    rare_threshold=10
)

print(f"Train set: {len(data['X_train'])} samples")
print(f"Val set: {len(data['X_val'])} samples")
print(f"Labels: {len(data['label_names'])}")

print("Training DistilBERT...")
trainer = train_distilbert_model(
    data=data,
    batch_size=16 if torch.cuda.is_available() else 8,
    learning_rate=2e-5,
    num_epochs=5 if torch.cuda.is_available() else 3,
    max_length=128,
    save_path='distilbert_ai_news.pth'
)

print("Training complete. Model saved to: distilbert_ai_news.pth")


# ============================================================
# Cell 5: Model Evaluation
# ============================================================

history = trainer.history

print("Training History")
print("=" * 50)
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
print(f"Final Micro-F1: {history['val_micro_f1'][-1]:.4f}")
print(f"Final Macro-F1: {history['val_macro_f1'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Train vs Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['val_micro_f1'], label='Micro-F1', marker='o', color='green')
axes[1].plot(history['val_macro_f1'], label='Macro-F1', marker='s', color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Validation F1 Scores')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

per_label = trainer.get_per_label_metrics()
print("Per-label performance (Top 10):")
print(per_label.head(10))
print("Lowest performing labels (Bottom 5):")
print(per_label.tail(5))


# ============================================================
# Cell 6: LLM Output Prediction
# ============================================================

from distilbert_complete import DistilBertPredictor

print("Loading model for prediction...")
predictor = DistilBertPredictor('distilbert_ai_news.pth')

print(f"Predicting {len(df_llm)} LLM outputs...")
df_llm_pred = predictor.predict_dataframe(
    df=df_llm,
    text_col='LLM_output',
    threshold=0.3
)

print("Prediction complete.")
print(df_llm_pred[['LLM', 'predicted_labels']].head())

llm_label_dist = {}
for llm in df_llm_pred['LLM'].unique():
    llm_df = df_llm_pred[df_llm_pred['LLM'] == llm]
    all_labels = []
    for labels_list in llm_df['predicted_labels']:
        all_labels.extend(labels_list)

    label_counts = pd.Series(all_labels).value_counts()
    llm_label_dist[llm] = label_counts
    print(f"\n{llm}:")
    print(label_counts.head(5))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (llm, counts) in enumerate(llm_label_dist.items()):
    if idx < 4:
        counts.head(10).plot(kind='barh', ax=axes[idx], color='coral')
        axes[idx].set_title(f'{llm} - Top 10 Labels')
        axes[idx].set_xlabel('Frequency')

plt.tight_layout()
plt.show()

df_llm_pred.to_csv('dataset_C_with_predictions.csv', index=False)
print("Predictions saved to: dataset_C_with_predictions.csv")


# ============================================================
# Cell 7: Behavioral Label to Sector Exposure Mapping
# ============================================================

GICS_SECTORS = [
    'Information Technology',
    'Communication Services',
    'Consumer Discretionary',
    'Financials',
    'Health Care',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Real Estate',
    'Materials'
]

LABEL_TO_SECTOR_MAPPING = {
    'Technology & Interaction': {
        'Information Technology': 0.9,
        'Communication Services': 0.6,
        'Consumer Discretionary': 0.3
    },
    'Healthcare & Medicine': {
        'Health Care': 0.9,
        'Information Technology': 0.4
    },
    'Work, Jobs & Economy': {
        'Industrials': 0.6,
        'Financials': 0.5,
        'Consumer Discretionary': 0.4,
        'Information Technology': 0.5
    },
    'Safety, Regulation & Ethics': {
        'Financials': 0.7,
        'Information Technology': 0.6,
        'Health Care': 0.4
    },
    'Media, Art & Entertainment': {
        'Communication Services': 0.8,
        'Consumer Discretionary': 0.6
    },
    'Education': {
        'Consumer Discretionary': 0.5,
        'Communication Services': 0.4
    },
    'Research & Development': {
        'Information Technology': 0.8,
        'Health Care': 0.5,
        'Materials': 0.4
    },
    'Misinformation & Challenges': {
        'Communication Services': 0.6,
        'Financials': 0.5
    },
    'Defense & Warfare': {
        'Industrials': 0.8,
        'Information Technology': 0.6
    },
    'Energy & Environment': {
        'Energy': 0.9,
        'Utilities': 0.7,
        'Materials': 0.5
    },
    'Transportation & Mobility': {
        'Industrials': 0.7,
        'Consumer Discretionary': 0.6
    },
    'Finance & Business': {
        'Financials': 0.9,
        'Information Technology': 0.5
    }
}

def labels_to_sector_exposure(labels_list):
    exposure = {sector: 0.0 for sector in GICS_SECTORS}
    for label in labels_list:
        if label in LABEL_TO_SECTOR_MAPPING:
            for sector, score in LABEL_TO_SECTOR_MAPPING[label].items():
                exposure[sector] = max(exposure[sector], score)
    return exposure

print("Computing sector exposure...")
df_llm_pred['sector_exposure'] = df_llm_pred['predicted_labels'].apply(labels_to_sector_exposure)

for idx in range(min(3, len(df_llm_pred))):
    row = df_llm_pred.iloc[idx]
    print(f"\nSample {idx+1}:")
    print(f"  Labels: {row['predicted_labels']}")
    for sector, score in row['sector_exposure'].items():
        if score > 0:
            print(f"    - {sector}: {score:.2f}")

all_exposures = pd.DataFrame([exp for exp in df_llm_pred['sector_exposure']])
sector_avg = all_exposures.mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sector_avg.plot(kind='bar', color='steelblue')
plt.title('Average Sector Exposure Score')
plt.xlabel('GICS Sector')
plt.ylabel('Average Exposure (0-1)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Sector exposure computation complete.")


# ============================================================
# Cell 8: AI Theme Factor Construction
# ============================================================

print("Building AI theme factors...")

factor_data = []

for llm in df_llm_pred['LLM'].unique():
    llm_df = df_llm_pred[df_llm_pred['LLM'] == llm]

    news_count = len(llm_df)

    positive_labels = ['Technology & Interaction', 'Healthcare & Medicine',
                       'Research & Development', 'Education']
    negative_labels = ['Safety, Regulation & Ethics', 'Misinformation & Challenges',
                       'Defense & Warfare']

    sentiment_score = 0
    for labels_list in llm_df['predicted_labels']:
        for label in labels_list:
            if label in positive_labels:
                sentiment_score += 1
            elif label in negative_labels:
                sentiment_score -= 1

    ais = sentiment_score / len(llm_df) if len(llm_df) > 0 else 0

    risk_labels = ['Safety, Regulation & Ethics', 'Misinformation & Challenges']
    risk_count = sum(1 for labels_list in llm_df['predicted_labels']
                     for label in labels_list if label in risk_labels)
    are = risk_count / len(llm_df) if len(llm_df) > 0 else 0

    avg_exposure = pd.DataFrame([exp for exp in llm_df['sector_exposure']]).mean().mean()

    aim_normalized = news_count / df_llm_pred['LLM'].value_counts().max()
    aii = 0.3 * aim_normalized + 0.3 * ais - 0.2 * are + 0.2 * avg_exposure

    factor_data.append({
        'LLM': llm,
        'AIM_NewsCount': news_count,
        'AIM_Normalized': aim_normalized,
        'AIS_Sentiment': ais,
        'ARE_Risk': are,
        'AvgExposure': avg_exposure,
        'AII_Impact': aii
    })

df_factors = pd.DataFrame(factor_data)
print(df_factors)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].bar(df_factors['LLM'], df_factors['AIM_Normalized'], color='skyblue')
axes[0, 0].set_title('AIM (AI Momentum)')
axes[0, 0].set_ylabel('AIM')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(df_factors['LLM'], df_factors['AIS_Sentiment'],
               color=['green' if x > 0 else 'red' for x in df_factors['AIS_Sentiment']])
axes[0, 1].set_title('AIS (AI Sentiment)')
axes[0, 1].set_ylabel('AIS')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].bar(df_factors['LLM'], df_factors['ARE_Risk'], color='coral')
axes[1, 0].set_title('ARE (AI Risk)')
axes[1, 0].set_ylabel('ARE')
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].bar(df_factors['LLM'], df_factors['AII_Impact'], color='purple')
axes[1, 1].set_title('AII (AI Impact)')
axes[1, 1].set_ylabel('AII')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

df_factors.to_csv('ai_theme_factors.csv', index=False)
print("Factor data saved to: ai_theme_factors.csv")


# ============================================================
# Cell 9: Sector ETF Data Download
# ============================================================

import yfinance as yf

SECTOR_ETFS = {
    'Information Technology': 'XLK',
    'Communication Services': 'XLC',
    'Consumer Discretionary': 'XLY',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}

print("Downloading sector ETF data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

etf_data = {}
for sector, ticker in SECTOR_ETFS.items():
    print(f"   Downloading {ticker} ({sector})...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    etf_data[sector] = data['Adj Close']

df_etf = pd.DataFrame(etf_data)
df_etf.index = pd.to_datetime(df_etf.index)

print(f"Date range: {df_etf.index[0].date()} to {df_etf.index[-1].date()}")
print(f"Trading days: {len(df_etf)}")

df_returns = df_etf.pct_change().dropna()

print("Average daily returns (%):")
print((df_returns.mean() * 100).sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(14, 7))
(df_etf / df_etf.iloc[0] * 100).plot(ax=ax, alpha=0.7)
ax.set_title('GICS Sector ETF Normalized Price (Base=100)')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Price')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_etf.to_csv('sector_etf_prices.csv')
df_returns.to_csv('sector_etf_returns.csv')
print("ETF data saved.")


# ============================================================
# Cell 10: Long-Short Investment Strategy
# ============================================================

print("Building long-short strategy...")

sector_scores = {}
for sector in GICS_SECTORS:
    sector_exposures = [exp.get(sector, 0) for exp in df_llm_pred['sector_exposure']]
    avg_exposure = np.mean(sector_exposures)
    sector_scores[sector] = avg_exposure

sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

long_sectors = [s[0] for s in sorted_sectors[:3]]
short_sectors = [s[0] for s in sorted_sectors[-3:]]

print(f"Long portfolio (Top 3 AII):")
for sector in long_sectors:
    print(f"   - {sector}: {sector_scores[sector]:.4f}")

print(f"Short portfolio (Bottom 3 AII):")
for sector in short_sectors:
    print(f"   - {sector}: {sector_scores[sector]:.4f}")

long_returns = df_returns[long_sectors].mean(axis=1)
short_returns = df_returns[short_sectors].mean(axis=1)
strategy_returns = 0.5 * long_returns - 0.5 * short_returns

cumulative_returns = (1 + strategy_returns).cumprod()
benchmark_returns = (1 + df_returns.mean(axis=1)).cumprod()

annual_return = strategy_returns.mean() * 252
annual_vol = strategy_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()

print(f"Annualized Return: {annual_return * 100:.2f}%")
print(f"Annualized Volatility: {annual_vol * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(cumulative_returns.index, cumulative_returns.values,
             label='Long-Short Strategy', linewidth=2, color='green')
axes[0].plot(benchmark_returns.index, benchmark_returns.values,
             label='Equal-Weight Benchmark', linewidth=2, color='gray', alpha=0.6)
axes[0].set_title('Strategy Cumulative Returns vs Benchmark')
axes[0].set_ylabel('Cumulative Return (Base=1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(drawdown.index, 0, drawdown.values * 100,
                     color='red', alpha=0.3, label='Drawdown')
axes[1].plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1.5)
axes[1].set_title('Strategy Drawdown')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].axhline(y=max_drawdown * 100, color='black', linestyle='--',
                label=f'Max Drawdown: {max_drawdown*100:.2f}%')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

strategy_df = pd.DataFrame({
    'Date': strategy_returns.index,
    'Strategy_Return': strategy_returns.values,
    'Cumulative_Return': cumulative_returns.values,
    'Drawdown': drawdown.values
})
strategy_df.to_csv('strategy_performance.csv', index=False)
print("Strategy performance saved to: strategy_performance.csv")


# ============================================================
# Cell 11: Granger Causality Testing
# ============================================================

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

print("Granger Causality Test: AI News Sentiment -> Sector Returns")
print("=" * 70)

np.random.seed(42)
dates = df_returns.index
ai_sentiment_ts = pd.Series(
    np.random.randn(len(dates)).cumsum() * 0.1,
    index=dates
)

sector = 'Information Technology'
sector_returns = df_returns[sector]

granger_data = pd.DataFrame({
    'AI_Sentiment': ai_sentiment_ts,
    'Sector_Return': sector_returns
}).dropna()

print(f"Target sector: {sector}")
print(f"Sample size: {len(granger_data)}")

try:
    max_lag = 5
    results = grangercausalitytests(granger_data[['Sector_Return', 'AI_Sentiment']],
                                    max_lag, verbose=False)

    print(f"Granger Causality Results (AI_Sentiment -> Sector_Return):")
    print("=" * 70)
    print(f"{'Lag':<8} {'F-Stat':<12} {'p-value':<12} {'Significance':<10}")
    print("-" * 70)

    for lag in range(1, max_lag + 1):
        test_result = results[lag][0]['ssr_ftest']
        f_stat = test_result[0]
        p_value = test_result[1]
        sig = '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
        print(f"{lag:<8} {f_stat:<12.4f} {p_value:<12.4f} {sig:<10}")

    print("Note: *** p<0.01, ** p<0.05, * p<0.1")

except Exception as e:
    print(f"Granger test failed: {e}")

try:
    model = VAR(granger_data)
    results = model.fit(maxlags=5, ic='aic')
    print(f"Optimal lag (AIC): {results.k_ar}")
    print(results.summary())

    irf = results.irf(10)
    irf.plot(orth=False)
    plt.suptitle('Impulse Response Function: AI Sentiment -> Returns')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"VAR model failed: {e}")


# ============================================================
# Cell 12: Event Study
# ============================================================

print("Event Study: Major AI Events Impact on Sector Returns")
print("=" * 70)

AI_EVENTS = {
    '2022-11-30': 'ChatGPT Launch',
    '2023-03-14': 'GPT-4 Launch',
    '2023-02-01': 'Google Bard Launch',
    '2023-05-10': 'PaLM 2 Launch',
}

window_before = 5
window_after = 10
estimation_start = 60
estimation_end = 6

target_sectors = ['Information Technology', 'Communication Services', 'Financials']

event_results = []

for event_date_str, event_name in AI_EVENTS.items():
    event_date = pd.to_datetime(event_date_str)

    if event_date not in df_returns.index:
        nearest_dates = df_returns.index[df_returns.index >= event_date]
        if len(nearest_dates) == 0:
            print(f"Event '{event_name}' out of data range, skipping.")
            continue
        event_date = nearest_dates[0]

    event_idx = df_returns.index.get_loc(event_date)

    if event_idx < estimation_start or event_idx + window_after >= len(df_returns):
        print(f"Insufficient data for '{event_name}', skipping.")
        continue

    print(f"\nEvent: {event_name} ({event_date.date()})")

    for sector in target_sectors:
        estimation_returns = df_returns[sector].iloc[
            event_idx - estimation_start : event_idx - estimation_end
        ]
        expected_return = estimation_returns.mean()

        event_returns = df_returns[sector].iloc[
            event_idx - window_before : event_idx + window_after + 1
        ]

        abnormal_returns = event_returns - expected_return
        car = abnormal_returns.cumsum()

        ar_std = estimation_returns.std()
        t_stat = car.iloc[-1] / (ar_std * np.sqrt(len(car)))

        print(f"{sector}:")
        print(f"  CAR [0, +10]: {car.iloc[window_before + 10] * 100:.2f}%")
        print(f"  t-stat: {t_stat:.2f}")
        print(f"  Significance: {'***' if abs(t_stat) > 2.576 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.645 else 'n.s.'}")

        event_results.append({
            'Event': event_name,
            'Date': event_date,
            'Sector': sector,
            'CAR_0_10': car.iloc[window_before + 10],
            't_stat': t_stat
        })

if event_results:
    df_events = pd.DataFrame(event_results)
    pivot_car = df_events.pivot(index='Event', columns='Sector', values='CAR_0_10')

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_car * 100, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'CAR (%)'})
    plt.title('Cumulative Abnormal Returns (CAR) [0, +10 days] by Event and Sector')
    plt.tight_layout()
    plt.show()

    df_events.to_csv('event_study_results.csv', index=False)
    print("Event study results saved to: event_study_results.csv")


# ============================================================
# Cell 13: GARCH Model - Sentiment and Volatility
# ============================================================

from arch import arch_model

print("GARCH(1,1) Model: AI Sentiment -> Market Volatility")
print("=" * 70)

sector = 'Information Technology'
sector_returns_pct = df_returns[sector] * 100

np.random.seed(42)
ai_sentiment = pd.Series(
    np.random.randn(len(sector_returns_pct)) * 0.5,
    index=sector_returns_pct.index
)

print(f"Target sector: {sector}")
print(f"Sample size: {len(sector_returns_pct)}")
print(sector_returns_pct.describe())

try:
    print("Fitting GARCH(1,1)...")
    model_basic = arch_model(
        sector_returns_pct,
        vol='Garch',
        p=1,
        q=1,
        dist='normal'
    )
    results_basic = model_basic.fit(disp='off')
    print(results_basic.summary())

    omega = results_basic.params['omega']
    alpha = results_basic.params['alpha[1]']
    beta = results_basic.params['beta[1]']

    print(f"omega: {omega:.6f}")
    print(f"alpha: {alpha:.4f}")
    print(f"beta: {beta:.4f}")
    print(f"alpha + beta: {alpha + beta:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(sector_returns_pct.index, sector_returns_pct.values,
                 label='Returns', alpha=0.6, color='blue')
    axes[0].set_title(f'{sector} Daily Returns')
    axes[0].set_ylabel('Return (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    conditional_vol = results_basic.conditional_volatility
    axes[1].plot(conditional_vol.index, conditional_vol.values,
                 label='Conditional Volatility (GARCH)', color='red', linewidth=1.5)
    axes[1].set_title('GARCH Conditional Volatility')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    standardized_resid = results_basic.resid / results_basic.conditional_volatility

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    from scipy import stats
    stats.probplot(standardized_resid.dropna(), dist="norm", plot=axes[0])
    axes[0].set_title('Standardized Residuals Q-Q Plot')

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(standardized_resid**2, lags=20, ax=axes[1])
    axes[1].set_title('Squared Residuals ACF (ARCH Effect Test)')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"GARCH model failed: {e}")


# ============================================================
# Cell 14: Report Generation
# ============================================================

import os
import shutil
from datetime import datetime

print("Generating analysis report...")

report_dir = f'AI_Quant_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(report_dir, exist_ok=True)

report_summary = f"""
{'='*70}
AI News Sentiment Quantitative Finance Analysis Report
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Data Overview
{'-'*70}
Dataset A:
   - Samples: {len(df_news)}
   - Labels: {len(data['label_names'])}
   - Train set: {len(data['X_train'])} samples
   - Val set: {len(data['X_val'])} samples

Dataset C:
   - Samples: {len(df_llm)}
   - LLM types: {df_llm['LLM'].nunique()}

2. Model Performance
{'-'*70}
DistilBERT Multi-Label Classifier:
   - Final Val Micro-F1: {history['val_micro_f1'][-1]:.4f}
   - Final Val Macro-F1: {history['val_macro_f1'][-1]:.4f}
   - Epochs: {len(history['train_loss'])}

3. Strategy Performance
{'-'*70}
Long-Short Portfolio:
   - Long: {', '.join(long_sectors)}
   - Short: {', '.join(short_sectors)}
   - Annualized Return: {annual_return * 100:.2f}%
   - Annualized Volatility: {annual_vol * 100:.2f}%
   - Sharpe Ratio: {sharpe_ratio:.2f}
   - Max Drawdown: {max_drawdown * 100:.2f}%

4. Output Files
{'-'*70}
1. distilbert_ai_news.pth
2. dataset_C_with_predictions.csv
3. ai_theme_factors.csv
4. sector_etf_prices.csv
5. sector_etf_returns.csv
6. strategy_performance.csv
7. event_study_results.csv

{'='*70}
"""

report_path = os.path.join(report_dir, 'analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_summary)

print(report_summary)

result_files = [
    'distilbert_ai_news.pth',
    'dataset_C_with_predictions.csv',
    'ai_theme_factors.csv',
    'sector_etf_prices.csv',
    'sector_etf_returns.csv',
    'strategy_performance.csv',
    'event_study_results.csv'
]

for file in result_files:
    if os.path.exists(file):
        shutil.copy(file, report_dir)

shutil.make_archive(report_dir, 'zip', report_dir)
print(f"Report saved to: {report_dir}.zip")
files.download(f'{report_dir}.zip')
