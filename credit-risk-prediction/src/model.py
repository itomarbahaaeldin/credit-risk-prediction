"""
Credit Risk Prediction - Model Training
========================================
LightGBM model training with cross-validation.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(train, test):
    """
    Encode categorical columns using LabelEncoder.
    
    Args:
        train: Training DataFrame
        test: Test DataFrame
        
    Returns:
        Encoded train and test DataFrames
    """
    le = LabelEncoder()
    
    for col in train.columns:
        if train[col].dtype == 'object':
            # Combine train and test for consistent encoding
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)
            
            # Fit on combined data
            le.fit(pd.concat([train[col], test[col]]))
            
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
    
    return train, test


def train_lightgbm_cv(X, y, X_test=None, n_splits=5, params=None):
    """
    Train LightGBM with Stratified K-Fold cross-validation.
    
    Args:
        X: Training features
        y: Target variable
        X_test: Test features (optional)
        n_splits: Number of CV folds
        params: LightGBM parameters (optional)
        
    Returns:
        oof_preds: Out-of-fold predictions
        test_preds: Test predictions (if X_test provided)
        scores: List of fold AUC scores
    """
    if params is None:
        params = {
            'n_estimators': 12000,
            'learning_rate': 0.005,
            'num_leaves': 96,
            'max_depth': 16,
            'colsample_bytree': 0.75,
            'subsample': 0.85,
            'feature_fraction': 0.75,
            'min_child_samples': 20,
            'lambda_l1': 1.0,
            'lambda_l2': 2.5,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    # Initialize arrays
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    scores = []
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize model
        model = lgb.LGBMClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=500)
            ]
        )
        
        # Predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        if X_test is not None:
            test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    # Overall score
    print(f"\n{'='*50}")
    print(f"Overall CV AUC: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
    print(f"{'='*50}")
    
    return oof_preds, test_preds, scores


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate predictions with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'F1': f1_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print("\n" + "="*40)
    print("Model Performance")
    print("="*40)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("="*40)


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from trained LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importances
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


if __name__ == "__main__":
    print("Credit Risk Model - Ready for training")
    print("Use train_lightgbm_cv() function with your data")
