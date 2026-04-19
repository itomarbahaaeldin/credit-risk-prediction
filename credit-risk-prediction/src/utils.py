"""
Credit Risk Prediction - Utilities
===================================
Data loading, preprocessing, and helper functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir):
    """
    Load all Home Credit dataset tables.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Dictionary of DataFrames
    """
    data_dir = Path(data_dir)
    
    tables = {
        'application_train': 'application_train.csv',
        'application_test': 'application_test.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'previous_application': 'previous_application.csv',
        'POS_CASH_balance': 'POS_CASH_balance.csv',
        'installments_payments': 'installments_payments.csv',
        'credit_card_balance': 'credit_card_balance.csv'
    }
    
    data = {}
    for name, filename in tables.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            data[name] = pd.read_csv(filepath)
            print(f"  Shape: {data[name].shape}")
        else:
            print(f"Warning: {filename} not found")
    
    return data


def reduce_memory_usage(df, verbose=True):
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Print memory reduction info
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def handle_missing_values(train, test, strategy='median'):
    """
    Handle missing values in train and test sets.
    
    Args:
        train: Training DataFrame
        test: Test DataFrame
        strategy: 'median', 'mean', or 'zero'
        
    Returns:
        Filled train and test DataFrames
    """
    if strategy == 'median':
        fill_values = train.median()
    elif strategy == 'mean':
        fill_values = train.mean()
    elif strategy == 'zero':
        fill_values = 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    train = train.fillna(fill_values)
    test = test.fillna(fill_values)
    
    return train, test


def create_submission(test_ids, predictions, filename='submission.csv'):
    """
    Create Kaggle submission file.
    
    Args:
        test_ids: SK_ID_CURR values
        predictions: Predicted probabilities
        filename: Output filename
    """
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"Shape: {submission.shape}")
    
    return submission


def check_target_distribution(y):
    """
    Check target variable distribution for class imbalance.
    
    Args:
        y: Target variable
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    print("\nTarget Distribution:")
    print("-" * 30)
    for label in sorted(counts.index):
        print(f"  Class {label}: {counts[label]:,} ({percentages[label]:.2f}%)")
    print("-" * 30)
    print(f"  Imbalance Ratio: 1:{counts[0]/counts[1]:.1f}")


if __name__ == "__main__":
    print("Credit Risk Utilities - Ready")
    print("Use load_data(data_dir) to load the dataset")
