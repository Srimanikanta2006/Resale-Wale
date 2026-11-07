"""
Comprehensive Model Evaluation & Statistical Testing Module
Follows ML Best Practices:
- Primary metrics: RMSE, MAE, R2, MAPE (Mean Absolute Percentage Error), Median AE
- Bootstrap 95% Confidence Intervals for error metrics
- Slice-based error analysis across Vehicle Age, Brand Tier, and Condition
- Residual distribution checks
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    Computes key regression metrics for vehicle valuation.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    
    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MedAE': round(medae, 2),
        'MAPE (%)': round(mape, 2),
        'R2': round(r2, 4)
    }

def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstraps=1000, alpha=0.05, random_seed=42):
    """
    Computes (1 - alpha)% bootstrap confidence interval for a metric.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    bootstrap_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        score = metric_fn(y_true_arr[indices], y_pred_arr[indices])
        bootstrap_scores.append(score)
        
    lower_bound = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return round(lower_bound, 2), round(upper_bound, 2)

def slice_based_error_analysis(df_eval, y_true_col='resale_price', y_pred_col='predicted_price'):
    """
    Evaluates model performance across key slices:
    - Vehicle Age brackets (<3 yrs, 3-7 yrs, 7-12 yrs, >12 yrs)
    - Brand Tiers (Economy, Mainstream, Luxury)
    - Condition tiers (Fair, Good, Very Good, Excellent)
    """
    slices = {}
    
    # 1. Age slices
    df = df_eval.copy()
    if 'vehicle_age' not in df.columns and 'model_year' in df.columns:
        df['vehicle_age'] = 2024 - df['model_year']
        
    age_bins = [-1, 3, 7, 12, 100]
    age_labels = ['< 3 Years (Newer)', '3-7 Years (Mid-Age)', '7-12 Years (Mature)', '> 12 Years (Older)']
    df['age_slice'] = pd.cut(df['vehicle_age'], bins=age_bins, labels=age_labels)
    
    age_perf = []
    for label in age_labels:
        subset = df[df['age_slice'] == label]
        if len(subset) > 0:
            m = calculate_metrics(subset[y_true_col], subset[y_pred_col])
            m['Slice'] = label
            m['Count'] = len(subset)
            age_perf.append(m)
    slices['Age_Bracket'] = pd.DataFrame(age_perf)
    
    # 2. Brand Tier slices
    if 'brand_tier' in df.columns:
        tier_perf = []
        for tier in ['Economy', 'Mainstream', 'Luxury']:
            subset = df[df['brand_tier'] == tier]
            if len(subset) > 0:
                m = calculate_metrics(subset[y_true_col], subset[y_pred_col])
                m['Slice'] = tier
                m['Count'] = len(subset)
                tier_perf.append(m)
        slices['Brand_Tier'] = pd.DataFrame(tier_perf)
        
    # 3. Condition slices
    if 'condition' in df.columns:
        cond_perf = []
        for cond in ['Fair', 'Good', 'Very Good', 'Excellent']:
            subset = df[df['condition'] == cond]
            if len(subset) > 0:
                m = calculate_metrics(subset[y_true_col], subset[y_pred_col])
                m['Slice'] = cond
                m['Count'] = len(subset)
                cond_perf.append(m)
        slices['Condition'] = pd.DataFrame(cond_perf)
        
    return slices

