"""
Credit Risk Prediction - Feature Engineering
=============================================
Functions for creating features from Home Credit dataset tables.
"""

import pandas as pd
import numpy as np


def agg_numeric(df, group_col, prefix):
    """
    Aggregate numeric columns by a grouping column.
    
    Args:
        df: DataFrame to aggregate
        group_col: Column to group by (e.g., 'SK_ID_CURR')
        prefix: Prefix for new column names
        
    Returns:
        Aggregated DataFrame with mean, sum, min, max for each numeric column
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the grouping column from numeric columns
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    # Define aggregation functions
    agg_funcs = ['mean', 'sum', 'min', 'max']
    
    # Perform aggregation
    agg_df = df.groupby(group_col)[numeric_cols].agg(agg_funcs)
    
    # Flatten column names
    agg_df.columns = [f'{prefix}_{col}_{func}' for col, func in agg_df.columns]
    
    # Reset index
    agg_df = agg_df.reset_index()
    
    return agg_df


def agg_categorical(df, group_col, prefix):
    """
    Aggregate categorical columns by counting unique values.
    
    Args:
        df: DataFrame to aggregate
        group_col: Column to group by
        prefix: Prefix for new column names
        
    Returns:
        Aggregated DataFrame with counts and nunique for categorical columns
    """
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(cat_cols) == 0:
        return pd.DataFrame({group_col: df[group_col].unique()})
    
    # Count aggregations
    agg_df = df.groupby(group_col).agg({col: ['count', 'nunique'] for col in cat_cols})
    
    # Flatten column names
    agg_df.columns = [f'{prefix}_{col}_{func}' for col, func in agg_df.columns]
    
    agg_df = agg_df.reset_index()
    
    return agg_df


def create_ratio_features(df):
    """
    Create ratio and interaction features from application data.
    
    Args:
        df: Application DataFrame
        
    Returns:
        DataFrame with new ratio features
    """
    df = df.copy()
    
    # Credit to annuity ratio (loan term indicator)
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    
    # Credit to income ratio (financial stress indicator)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    # Goods price to credit ratio (down payment indicator)
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    
    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # Age in years
    df['AGE'] = -df['DAYS_BIRTH'] / 365
    
    # Employment to age ratio
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    
    # Credit minus goods price (loan overhead)
    df['CREDIT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    
    # Payment rate (annuity / credit)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    
    # Age binned into quintiles
    df['AGE_BINNED'] = pd.qcut(df['AGE'], q=5, labels=False, duplicates='drop')
    
    return df


def process_bureau(bureau, bureau_balance):
    """
    Process bureau and bureau_balance tables.
    
    Args:
        bureau: Bureau DataFrame
        bureau_balance: Bureau balance DataFrame
        
    Returns:
        Aggregated bureau features by SK_ID_CURR
    """
    # Aggregate bureau_balance by SK_ID_BUREAU
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'STATUS': lambda x: (x == 'C').sum()  # Count closed status
    })
    bb_agg.columns = ['BB_MONTHS_MIN', 'BB_MONTHS_MAX', 'BB_COUNT', 'BB_CLOSED_COUNT']
    bb_agg = bb_agg.reset_index()
    
    # Merge with bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Aggregate bureau by SK_ID_CURR
    bureau_agg = agg_numeric(bureau, 'SK_ID_CURR', 'BUREAU')
    
    # Count number of bureau records per client
    bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_COUNT')
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
    
    return bureau_agg


def process_previous_application(prev_app):
    """
    Process previous application table.
    
    Args:
        prev_app: Previous application DataFrame
        
    Returns:
        Aggregated previous application features by SK_ID_CURR
    """
    # Create additional features
    prev_app['APP_CREDIT_RATIO'] = prev_app['AMT_APPLICATION'] / (prev_app['AMT_CREDIT'] + 1)
    prev_app['CREDIT_GOODS_RATIO'] = prev_app['AMT_CREDIT'] / (prev_app['AMT_GOODS_PRICE'] + 1)
    
    # Aggregate by SK_ID_CURR
    prev_agg = agg_numeric(prev_app, 'SK_ID_CURR', 'PREV')
    
    # Count previous applications
    prev_counts = prev_app.groupby('SK_ID_CURR').size().reset_index(name='PREV_APP_COUNT')
    prev_agg = prev_agg.merge(prev_counts, on='SK_ID_CURR', how='left')
    
    # Count approved/refused applications
    prev_status = prev_app.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
    prev_status.columns = [f'PREV_STATUS_{col}' for col in prev_status.columns]
    prev_status = prev_status.reset_index()
    prev_agg = prev_agg.merge(prev_status, on='SK_ID_CURR', how='left')
    
    return prev_agg


def process_pos_cash(pos):
    """
    Process POS_CASH_balance table.
    
    Args:
        pos: POS_CASH_balance DataFrame
        
    Returns:
        Aggregated POS features by SK_ID_CURR
    """
    pos_agg = agg_numeric(pos, 'SK_ID_CURR', 'POS')
    
    # Count POS records
    pos_counts = pos.groupby('SK_ID_CURR').size().reset_index(name='POS_COUNT')
    pos_agg = pos_agg.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    return pos_agg


def process_installments(installments):
    """
    Process installments_payments table.
    
    Args:
        installments: Installments payments DataFrame
        
    Returns:
        Aggregated installment features by SK_ID_CURR
    """
    # Calculate payment difference
    installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
    installments['PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 1)
    installments['DAYS_DIFF'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    
    # Aggregate by SK_ID_CURR
    inst_agg = agg_numeric(installments, 'SK_ID_CURR', 'INST')
    
    return inst_agg


def process_credit_card(cc):
    """
    Process credit_card_balance table.
    
    Args:
        cc: Credit card balance DataFrame
        
    Returns:
        Aggregated credit card features by SK_ID_CURR
    """
    # Calculate utilization
    cc['CREDIT_UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    
    # Aggregate by SK_ID_CURR
    cc_agg = agg_numeric(cc, 'SK_ID_CURR', 'CC')
    
    # Count credit card records
    cc_counts = cc.groupby('SK_ID_CURR').size().reset_index(name='CC_COUNT')
    cc_agg = cc_agg.merge(cc_counts, on='SK_ID_CURR', how='left')
    
    return cc_agg


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing feature engineering functions...")
    
    # Create sample application data
    sample_app = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'AMT_CREDIT': [100000, 200000, 150000],
        'AMT_ANNUITY': [10000, 15000, 12000],
        'AMT_INCOME_TOTAL': [50000, 80000, 60000],
        'AMT_GOODS_PRICE': [90000, 180000, 140000],
        'CNT_FAM_MEMBERS': [2, 4, 3],
        'DAYS_BIRTH': [-10000, -15000, -12000],
        'DAYS_EMPLOYED': [-2000, -5000, -3000]
    })
    
    # Test ratio features
    result = create_ratio_features(sample_app)
    print(f"Ratio features created: {[col for col in result.columns if col not in sample_app.columns]}")
    
    print("\nAll tests passed!")
