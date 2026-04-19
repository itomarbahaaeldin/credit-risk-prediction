"""
Credit Risk Prediction Package
===============================
Predicting loan default risk using LightGBM with advanced feature engineering.
"""

from .features import (
    agg_numeric,
    agg_categorical,
    create_ratio_features,
    process_bureau,
    process_previous_application,
    process_pos_cash,
    process_installments,
    process_credit_card
)

from .model import (
    encode_categoricals,
    train_lightgbm_cv,
    evaluate_predictions,
    get_feature_importance
)

from .utils import (
    load_data,
    reduce_memory_usage,
    handle_missing_values,
    create_submission,
    check_target_distribution
)

__version__ = '1.0.0'
__author__ = 'Omar Bahaaeldin'
