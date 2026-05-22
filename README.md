<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=180&section=header&text=Credit%20Risk%20Prediction&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=Kaggle%20Home%20Credit%20%7C%20AUC%200.787%20%7C%20LightGBM%20%2B%20Feature%20Engineering&descAlignY=58&descSize=14&animation=fadeIn" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GPU-9333ea?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.787-22c55e?style=flat-square)
![CV](https://img.shields.io/badge/Cross--Val-5--Fold%20Stratified-0f3460?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Home%20Credit-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

</div>

<br/>

## 📋 Overview

A complete solution for the **Kaggle Home Credit Default Risk** competition — predicting whether a loan applicant will default based on their financial history, demographics, and credit bureau data.

The solution joins **7 data tables** (57M+ rows combined), engineers 250+ features, and trains a GPU-accelerated LightGBM classifier evaluated with 5-fold stratified cross-validation.

<br/>

## 📊 Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.787** |
| Accuracy | 78.6% |
| F1-Score | 0.316 |

*5-Fold Stratified Cross-Validation · Class-balanced training*

> AUC-ROC is the primary competition metric. A score of 0.787 represents strong performance on this notoriously imbalanced dataset (only ~8% defaults).

<br/>

## 🏗️ Pipeline

```
7 Raw Data Sources (57M+ rows total)
        │
        ▼
┌─────────────────────────────────┐
│       Feature Engineering       │
│  Aggregations per client:        │
│  mean · sum · min · max · std   │
│  count · nunique (categoricals) │
│                                  │
│  Ratio & interaction features:   │
│  CREDIT/INCOME · ANNUITY/CREDIT  │
│  EMPLOYED_AGE_RATIO · PAYMENT_RATE│
└─────────────────┬───────────────┘
                  │
                  ▼
        500+ engineered features
                  │
                  ▼
     LightGBM feature importance
                  │
                  ▼
        Top 252 features selected
                  │
                  ▼
┌─────────────────────────────────┐
│   LightGBM Classifier (GPU)     │
│   5-Fold Stratified CV          │
│   Early Stopping (patience=200) │
└─────────────────────────────────┘
                  │
                  ▼
          AUC-ROC: 0.787
```

<br/>

## 📁 Dataset

| Table | Rows | Description |
|-------|------|-------------|
| application_train | 307,511 | Main training data with TARGET label |
| application_test | 48,744 | Test data for submission |
| bureau | 1,716,428 | Previous credits from other institutions |
| bureau_balance | 27,299,925 | Monthly balances of previous credits |
| previous_application | 1,670,214 | Previous Home Credit applications |
| POS_CASH_balance | 10,001,358 | Monthly POS/cash loan balances |
| credit_card_balance | 3,840,312 | Monthly credit card balances |
| installments_payments | 13,605,401 | Historical repayment records |

Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).

<br/>

## 🔍 Feature Engineering

### Aggregations (per client, per auxiliary table)
```python
# For each table joined on SK_ID_CURR:
 agg_funcs = ["mean", "sum", "min", "max"]    # numerical
cat_funcs  = ["count", "nunique"]                     # categorical
```

### Key Ratio & Interaction Features
```python
# Financial stress indicators
CREDIT_ANNUITY_RATIO  = AMT_CREDIT / AMT_ANNUITY
CREDIT_INCOME_RATIO   = AMT_CREDIT / AMT_INCOME_TOTAL
GOODS_CREDIT_RATIO    = AMT_GOODS_PRICE / AMT_CREDIT

# Demographics
INCOME_PER_PERSON     = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
AGE_YEARS             = -DAYS_BIRTH / 365
EMPLOYED_AGE_RATIO    = DAYS_EMPLOYED / DAYS_BIRTH

# Payment behaviour
PAYMENT_RATE          = AMT_ANNUITY / AMT_CREDIT
```

### Top Predictive Features
1. External credit bureau scores (`EXT_SOURCE_1/2/3`)
2. Credit-to-income ratio
3. Previous application history (number of rejections)
4. Age at application
5. Employment duration relative to age

<br/>

## ⚙️ Model Configuration

```python
model = lgb.LGBMClassifier(
    n_estimators     = 12000,
    learning_rate    = 0.005,
    num_leaves       = 96,
    max_depth        = 16,
    colsample_bytree = 0.75,
    subsample        = 0.85,
    feature_fraction = 0.75,
    min_child_samples= 20,
    lambda_l1        = 1.0,
    lambda_l2        = 2.5,
    class_weight     = "balanced",   # handles 8% default rate
    device           = "gpu"
)
```

**Training strategy:**
- 5-fold stratified cross-validation (preserves 8:92 class ratio per fold)
- Early stopping with patience=200 rounds
- GPU acceleration for the 12,000-estimator ensemble

<br/>

## 🚀 Quick Start

```bash
git clone https://github.com/itomarbahaaeldin/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
```

1. Download data from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place in `data/`
2. Open `notebooks/Credit_Risk_Analysis.ipynb`
3. Run all cells — feature engineering → training → evaluation

Or use the modular `src/` scripts:
```bash
python src/features.py    # Build feature matrix
python src/model.py       # Train and evaluate
```

<br/>

## 📁 Project Structure

```
credit-risk-prediction/
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb    # Full pipeline notebook
├── src/
│   ├── features.py                   # Feature engineering functions
│   ├── model.py                      # LightGBM training & CV
│   └── utils.py                      # Helpers, metrics, plotting
├── requirements.txt
├── LICENSE
└── README.md
```

<br/>

## 💡 Key Insights

1. **Class imbalance is the core challenge** — `class_weight='balanced'` is essential; without it F1 collapses
2. **External scores dominate** — `EXT_SOURCE_2` alone is the single best predictor
3. **Aggregations add massive signal** — joining all 7 tables and aggregating doubles AUC over application data alone
4. **Feature selection matters** — pruning from 500+ to top 252 reduces overfitting without sacrificing AUC
5. **LightGBM beats XGBoost here** — faster training on GPU with comparable accuracy

<br/>

## 🔮 Future Work

- [ ] SHAP values for full model interpretability
- [ ] Optuna hyperparameter optimisation
- [ ] Ensemble: LightGBM + XGBoost + CatBoost
- [ ] Neural network baseline (TabNet, FT-Transformer)
- [ ] Fairness analysis across demographic groups

<br/>

## 👨‍💻 Author

**Omar Bahaa Eldin**

[![Portfolio](https://img.shields.io/badge/Portfolio-000?style=flat-square&logo=vercel&logoColor=white)](https://itomarbahaaeldin.github.io/omar-bahaa-portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omar-bahaaeldin10)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:itomarbahaaeldin@gmail.com)

<br/>

## 📄 License

MIT © [Omar Bahaa Eldin](https://github.com/itomarbahaaeldin)

<div align="center">
<br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=100&section=footer" width="100%"/>
</div>
