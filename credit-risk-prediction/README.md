# 💳 Credit Risk Default Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-GPU-9cf)](https://lightgbm.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/home-credit-default-risk)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicting loan default risk using LightGBM with advanced feature engineering on the Home Credit dataset**

## 🎯 Project Overview

This project tackles the **Home Credit Default Risk** Kaggle competition, predicting whether a loan applicant will default on their loan. The solution uses **LightGBM with GPU acceleration** and extensive feature engineering from 7 data sources.

## 📊 Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.787** |
| **Accuracy** | 78.6% |
| **F1-Score** | 0.316 |

*5-Fold Stratified Cross-Validation Results*

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (7 tables)                  │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ application  │   bureau     │   previous   │  credit_card  │
│   _train     │   _balance   │ _application │   _balance    │
├──────────────┼──────────────┼──────────────┼───────────────┤
│  POS_CASH    │ installments │   bureau     │               │
│  _balance    │  _payments   │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   Feature Engineering    │
              │  • Aggregations (mean,   │
              │    sum, min, max, count) │
              │  • Ratio features        │
              │  • Interaction features  │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   LightGBM Classifier    │
              │  • GPU Acceleration      │
              │  • 5-Fold Stratified CV  │
              │  • Early Stopping        │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   AUC-ROC: 0.787         │
              └──────────────────────────┘
```

## 📁 Project Structure

```
credit-risk-prediction/
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb    # Complete analysis notebook
├── src/
│   ├── features.py                   # Feature engineering functions
│   ├── model.py                      # LightGBM model training
│   └── utils.py                      # Utility functions
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Installation

```bash
git clone https://github.com/itomarbahaaeldin/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
```

## 📈 Dataset

The dataset is from the [Kaggle Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk/data).

| Table | Rows | Description |
|-------|------|-------------|
| application_train | 307,511 | Main training data with TARGET |
| application_test | 48,744 | Test data for predictions |
| bureau | 1,716,428 | Client's previous credits from other institutions |
| bureau_balance | 27,299,925 | Monthly balances of previous credits |
| previous_application | 1,670,214 | Previous Home Credit loan applications |
| POS_CASH_balance | 10,001,358 | Monthly balance of POS/cash loans |
| credit_card_balance | 3,840,312 | Monthly credit card balance |
| installments_payments | 13,605,401 | Repayment history |

## 🔍 Feature Engineering

### Aggregation Features
For each auxiliary table, we compute aggregations per client:
- **Numerical**: mean, sum, min, max, std
- **Categorical**: count, nunique

### Ratio & Interaction Features
```python
# Financial stress indicators
CREDIT_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY
CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
GOODS_CREDIT_RATIO = AMT_GOODS_PRICE / AMT_CREDIT

# Demographics
INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
AGE = -DAYS_BIRTH / 365
EMPLOYED_AGE_RATIO = DAYS_EMPLOYED / DAYS_BIRTH

# Payment behavior
PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
```

### Feature Selection
- Used LightGBM feature importance
- Selected top 252 features from 500+ engineered features

## 🚀 Model Configuration

```python
model = lgb.LGBMClassifier(
    n_estimators=12000,
    learning_rate=0.005,
    num_leaves=96,
    max_depth=16,
    colsample_bytree=0.75,
    subsample=0.85,
    feature_fraction=0.75,
    min_child_samples=20,
    lambda_l1=1.0,
    lambda_l2=2.5,
    class_weight='balanced',
    device='gpu'
)
```

### Training Strategy
- **5-Fold Stratified Cross-Validation** for robust evaluation
- **Early Stopping** (patience=200) to prevent overfitting
- **GPU Acceleration** for faster training

## 📉 Key Insights

1. **Class Imbalance**: Only ~8% of loans default → used `class_weight='balanced'`
2. **Missing Values**: Handled with median imputation
3. **Top Features**: 
   - External credit bureau scores
   - Credit/income ratios
   - Previous application history
4. **Feature Engineering Impact**: Aggregated features from auxiliary tables significantly improved AUC

## 💡 Lessons Learned

- Combining raw data with engineered features leads to better predictive power
- Handling missing values and feature selection is crucial for model stability
- LightGBM with proper tuning achieves strong AUC scores for imbalanced classification

## 🔮 Future Improvements

- [ ] Try XGBoost and CatBoost ensemble
- [ ] Add more domain-specific features
- [ ] Implement SHAP for model interpretability
- [ ] Experiment with neural network approaches
- [ ] Add Optuna for hyperparameter optimization

## 👨‍💻 Author

**Omar Bahaaeldin**  
AI & Data Science Graduate | ESLSCA University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/omar-bahaaeldin10)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/itomarbahaaeldin)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if you found it helpful!**
