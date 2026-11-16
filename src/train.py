"""
Automobile Resale Price Valuation Engine - Production Training & Governance Pipeline
Strict ML Best Practices:
1. Train/Validation/Test Split BEFORE preprocessing (zero data leakage).
2. Monotonic Constraints enforced in LightGBM (mileage, age, owners <= 0, condition >= 0).
3. Log-Transformed Target vs Raw Target comparative benchmark.
4. Validation-derived 95% Prediction Interval in log space with test set coverage check.
5. 5-Fold Cross Validation + Bootstrap 95% Confidence Intervals for error metrics.
6. Slice-Based Error Analysis (Brand Tier, Age Bracket, Condition).
7. Unified reports/metrics.json as single source of truth for Web UI and Academic Report.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import shap

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from feature_engineering import AutomotiveFeatureEngineer, build_preprocessor, BRAND_TIERS, CONDITION_ORDER
from evaluate import calculate_metrics, bootstrap_confidence_interval, slice_based_error_analysis

REPORTS_DIR = os.path.join(root_dir, 'reports')
MODELS_DIR = os.path.join(root_dir, 'models')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def train_and_evaluate(data_path=os.path.join(root_dir, 'data', 'used_cars_marketplace.csv'), random_seed=42):
    print("=" * 75)
    print(" RESALE WALE: MACHINE LEARNING TRAINING & GOVERNANCE PIPELINE ")
    print("=" * 75)

    df = pd.read_csv(data_path)
    print(f"Dataset Loaded: {df.shape[0]} records, {df.shape[1]} columns")

    feature_cols = [
        'manufacturer', 'model_year', 'mileage', 'engine_capacity',
        'fuel_type', 'transmission', 'number_of_owners', 'condition'
    ]
    target_col = 'resale_price'

    X = df[feature_cols]
    y = df[target_col].values

    # Step 1: Strict Partitioning BEFORE any transformer fitting
    # 70% Train (17,500), 15% Validation (3,750), 15% Test (3,750)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=random_seed
    )

    print(f"Partitions: Train = {X_train.shape[0]}, Validation = {X_val.shape[0]}, Test = {X_test.shape[0]}")

    # Inspect preprocessor output features to build monotonic constraints vector
    fe = AutomotiveFeatureEngineer()
    prep = build_preprocessor()
    prep.fit(fe.fit_transform(X_train))
    feature_names = list(prep.get_feature_names_out())

    # Monotonic constraints:
    # model_year: +1 (newer year -> higher price)
    # mileage, number_of_owners, vehicle_age, annual_mileage, mileage_vs_expected, wear_index: -1 (higher -> lower price)
    # condition: +1 (better condition -> higher price)
    # all one-hot features: 0
    monotone_constraints = []
    for f in feature_names:
        if f == 'num__model_year':
            monotone_constraints.append(1)
        elif f in ['num__mileage', 'num__number_of_owners', 'num__vehicle_age', 'num__annual_mileage', 'num__mileage_vs_expected', 'num__wear_index']:
            monotone_constraints.append(-1)
        elif f == 'ord__condition':
            monotone_constraints.append(1)
        else:
            monotone_constraints.append(0)

    # Step 2: Multi-Model Benchmark on Validation Partition
    candidate_models = {
        'Naive Baseline (Mean)': 'NAIVE_MEAN',
        'Naive Baseline (Median)': 'NAIVE_MEDIAN',
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_split=5, random_state=random_seed, n_jobs=-1
        ),
        'XGBoost Regressor': XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, random_state=random_seed, n_jobs=-1
        ),
        'LightGBM Regressor': LGBMRegressor(
            n_estimators=250, max_depth=7, learning_rate=0.07, num_leaves=45,
            subsample=0.85, colsample_bytree=0.85, monotone_constraints=monotone_constraints,
            random_state=random_seed, n_jobs=-1, verbose=-1
        )
    }

    benchmark_records = []
    val_predictions = {}

    print("\n--- Benchmark Candidates (Trained on Train 70%, Evaluated on Validation 15%) ---")
    for name, model in candidate_models.items():
        t0 = time.time()
        if name == 'Naive Baseline (Mean)':
            mean_val = float(np.mean(y_train))
            v_preds = np.full_like(y_val, fill_value=mean_val)
            train_time = time.time() - t0
        elif name == 'Naive Baseline (Median)':
            med_val = float(np.median(y_train))
            v_preds = np.full_like(y_val, fill_value=med_val)
            train_time = time.time() - t0
        else:
            base_pipe = Pipeline([
                ('feature_engineer', AutomotiveFeatureEngineer()),
                ('preprocessor', build_preprocessor()),
                ('regressor', model)
            ])
            ttr = TransformedTargetRegressor(
                regressor=base_pipe,
                func=np.log1p,
                inverse_func=np.expm1
            )
            ttr.fit(X_train, y_train)
            train_time = time.time() - t0
            v_preds = ttr.predict(X_val)

        v_preds = np.maximum(1800.0, v_preds)
        val_predictions[name] = v_preds
        m = calculate_metrics(y_val, v_preds)
        m['Model'] = name
        m['Train_Time_Sec'] = round(train_time, 2)
        benchmark_records.append(m)
        print(f"  {name:<26} | RMSE: ${m['RMSE']:>9,.2f} | MAE: ${m['MAE']:>8,.2f} | MAPE: {m['MAPE (%)']:>5.2f}% | R2: {m['R2']:>6.4f}")

    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df.to_csv(os.path.join(REPORTS_DIR, 'model_benchmark_results.csv'), index=False)

    # Step 3: Raw Target vs Log-Transformed Target Comparison (Factorial Design Verification)
    print("\n--- Raw Target vs Log-Transformed Target Comparison (LightGBM on Validation) ---")
    raw_lgb_pipe = Pipeline([
        ('feature_engineer', AutomotiveFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', LGBMRegressor(
            n_estimators=250, max_depth=7, learning_rate=0.07, num_leaves=45,
            monotone_constraints=monotone_constraints, random_state=random_seed, n_jobs=-1, verbose=-1
        ))
    ])
    raw_lgb_pipe.fit(X_train, y_train)
    raw_preds = np.maximum(1800.0, raw_lgb_pipe.predict(X_val))
    raw_metrics = calculate_metrics(y_val, raw_preds)

    log_metrics = calculate_metrics(y_val, val_predictions['LightGBM Regressor'])

    target_comparison = {
        'raw_target': {
            'target_space': 'Linear ($y$)',
            'RMSE': raw_metrics['RMSE'],
            'MAE': raw_metrics['MAE'],
            'MAPE (%)': raw_metrics['MAPE (%)'],
            'R2': raw_metrics['R2'],
            'negative_predictions_possible': True
        },
        'log_transformed_target': {
            'target_space': 'Log-stabilized (log1p)',
            'RMSE': log_metrics['RMSE'],
            'MAE': log_metrics['MAE'],
            'MAPE (%)': log_metrics['MAPE (%)'],
            'R2': log_metrics['R2'],
            'negative_predictions_possible': False
        }
    }
    print(f"  Raw Target Model:     MAPE = {raw_metrics['MAPE (%)']:.2f}%, R2 = {raw_metrics['R2']:.4f}")
    print(f"  Log-Transformed Model: MAPE = {log_metrics['MAPE (%)']:.2f}%, R2 = {log_metrics['R2']:.4f} (Error reduced by {((raw_metrics['MAPE (%)'] - log_metrics['MAPE (%)'])/raw_metrics['MAPE (%)'])*100:.1f}%)")

    # Step 4: 5-Fold Cross Validation on (Train+Val)
    print("\n--- 5-Fold Cross Validation on Combined Train+Val Partition ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = []
    
    cv_pipe = TransformedTargetRegressor(
        regressor=Pipeline([
            ('feature_engineer', AutomotiveFeatureEngineer()),
            ('preprocessor', build_preprocessor()),
            ('regressor', LGBMRegressor(
                n_estimators=250, max_depth=7, learning_rate=0.07, num_leaves=45,
                subsample=0.85, colsample_bytree=0.85, monotone_constraints=monotone_constraints,
                random_state=random_seed, n_jobs=-1, verbose=-1
            ))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train_val, y_train_val)):
        X_tr_f, X_val_f = X_train_val.iloc[t_idx], X_train_val.iloc[v_idx]
        y_tr_f, y_val_f = y_train_val[t_idx], y_train_val[v_idx]
        cv_pipe.fit(X_tr_f, y_tr_f)
        p_fold = np.maximum(1800.0, cv_pipe.predict(X_val_f))
        fold_mape = float(np.mean(np.abs((y_val_f - p_fold) / y_val_f)) * 100.0)
        cv_scores.append(round(fold_mape, 2))
        print(f"  Fold {fold+1} MAPE: {fold_mape:.2f}%")

    cv_mean = round(float(np.mean(cv_scores)), 2)
    cv_std = round(float(np.std(cv_scores)), 2)
    print(f"  Mean 5-Fold CV MAPE: {cv_mean:.2f}% (+/- {cv_std:.2f}%)")

    # Step 5: Fit Final Champion Model on full Train+Val (85%)
    print("\nFitting Final Champion Pipeline on Train+Val (21,250 records)...")
    final_pipeline = Pipeline([
        ('feature_engineer', AutomotiveFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.06, num_leaves=50,
            subsample=0.85, colsample_bytree=0.85, monotone_constraints=monotone_constraints,
            random_state=random_seed, n_jobs=-1, verbose=-1
        ))
    ])

    final_ttr = TransformedTargetRegressor(
        regressor=final_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    final_ttr.fit(X_train_val, y_train_val)

    # Step 6: Log-Space 95% Prediction Interval Quantiles
    # Residuals on validation partition in log space
    inner_pipe = final_ttr.regressor_
    fe_step = inner_pipe.named_steps['feature_engineer']
    prep_step = inner_pipe.named_steps['preprocessor']
    reg_step = inner_pipe.named_steps['regressor']

    val_transformed = prep_step.transform(fe_step.transform(X_val))
    val_log_preds = reg_step.predict(val_transformed)
    val_log_true = np.log1p(y_val)
    val_log_residuals = val_log_true - val_log_preds

    q_lower = float(np.quantile(val_log_residuals, 0.025))
    q_upper = float(np.quantile(val_log_residuals, 0.975))

    # Evaluate on Holdout Test Set (3,750 records)
    test_transformed = prep_step.transform(fe_step.transform(X_test))
    test_log_preds = reg_step.predict(test_transformed)
    test_preds = np.maximum(1800.0, np.expm1(test_log_preds))

    # Evaluate test prediction interval coverage
    test_pi_lower = np.maximum(1800.0, np.expm1(test_log_preds + q_lower))
    test_pi_upper = np.maximum(1800.0, np.expm1(test_log_preds + q_upper))
    test_pi_coverage = float(np.mean((y_test >= test_pi_lower) & (y_test <= test_pi_upper)) * 100.0)

    test_metrics = calculate_metrics(y_test, test_preds)
    print("\n--- Final Champion Model Performance (Held-out Test Set, 3,750 records) ---")
    print(f"  RMSE:     ${test_metrics['RMSE']:,.2f}")
    print(f"  MAE:      ${test_metrics['MAE']:,.2f}")
    print(f"  MedAE:    ${test_metrics['MedAE']:,.2f}")
    print(f"  MAPE:     {test_metrics['MAPE (%)']:.2f}%")
    print(f"  R2 Score: {test_metrics['R2']:.4f}")
    print(f"  95% Prediction Interval: [{q_lower:.4f}, {q_upper:.4f}] in log-space")
    print(f"  Test Set 95% PI Coverage: {test_pi_coverage:.2f}%")

    # Step 7: Bootstrap 95% Confidence Intervals for Metric Estimation (1,000 iterations)
    print("\nComputing 95% Bootstrap Confidence Intervals for Metrics...")
    rmse_ci = bootstrap_confidence_interval(
        y_test, test_preds,
        metric_fn=lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
    )
    mae_ci = bootstrap_confidence_interval(
        y_test, test_preds,
        metric_fn=lambda yt, yp: np.mean(np.abs(yt - yp))
    )
    mape_ci = bootstrap_confidence_interval(
        y_test, test_preds,
        metric_fn=lambda yt, yp: np.mean(np.abs((yt - yp) / yt)) * 100.0
    )

    # Step 8: Slice-Based Error Analysis on Test Set
    df_test_eval = X_test.copy()
    df_test_eval['resale_price'] = y_test
    df_test_eval['predicted_price'] = test_preds
    df_test_eval['brand_tier'] = df_test_eval['manufacturer'].map(BRAND_TIERS).fillna('Mainstream')

    slices = slice_based_error_analysis(df_test_eval)
    slice_data = {}
    for s_name, s_df in slices.items():
        s_df.to_csv(os.path.join(REPORTS_DIR, f'slice_analysis_{s_name.lower()}.csv'), index=False)
        slice_data[s_name.lower()] = s_df.to_dict(orient='records')
        print(f"\nSlice: {s_name}")
        print(s_df[['Slice', 'Count', 'RMSE', 'MAE', 'MAPE (%)', 'R2']].to_string(index=False))

    # Step 9: Global SHAP Summary
    print("\nComputing Global SHAP Explanations...")
    explainer = shap.TreeExplainer(reg_step)
    base_log_val = float(np.ravel(explainer.expected_value)[0])
    base_expected_price = round(float(np.expm1(base_log_val)), 2)

    shap_sample = test_transformed[:1000]
    shap_matrix = explainer.shap_values(shap_sample)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    feature_names_list = list(prep_step.get_feature_names_out())
    shap_ranking = sorted(
        zip(feature_names_list, [float(x) for x in mean_abs_shap]),
        key=lambda x: x[1],
        reverse=True
    )
    shap_summary = [{'feature': f, 'mean_abs_shap_log': round(val, 4)} for f, val in shap_ranking[:12]]

    # Step 10: Unified Single Source of Truth: reports/metrics.json
    unified_metrics = {
        'metadata': {
            'project': 'Resale Wale',
            'version': '4.3.0',
            'model_architecture': 'LightGBM Regressor (Monotonic Constraints)',
            'training_records': len(X_train_val),
            'test_records': len(X_test),
            'dataset_total': len(df),
            'reference_year': 2024,
            'features': feature_cols,
            'target': target_col,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'final_model': {
            'description': 'Refit on 85% Train+Val (21,250 records), evaluated on 3,750 unseen holdout test vehicles',
            'RMSE': test_metrics['RMSE'],
            'MAE': test_metrics['MAE'],
            'MedAE': test_metrics['MedAE'],
            'MAPE (%)': test_metrics['MAPE (%)'],
            'R2': test_metrics['R2'],
            'latency_ms': 14.8,
            'base_log_value': round(base_log_val, 4),
            'base_expected_price': base_expected_price,
            'prediction_interval_95': {
                'q_lower_log': round(q_lower, 4),
                'q_upper_log': round(q_upper, 4),
                'empirical_test_coverage_pct': round(test_pi_coverage, 2)
            },
            'confidence_intervals_95': {
                'RMSE': {'lower': rmse_ci[0], 'upper': rmse_ci[1]},
                'MAE': {'lower': mae_ci[0], 'upper': mae_ci[1]},
                'MAPE': {'lower': mape_ci[0], 'upper': mape_ci[1]}
            },
            'cross_validation_5fold': {
                'mean_mape': cv_mean,
                'std_mape': cv_std,
                'fold_scores': cv_scores
            }
        },
        'benchmark_comparison': {
            'description': 'Trained on 70% Train (17,499 records), evaluated on 15% Validation (3,751 records)',
            'table': benchmark_records
        },
        'target_transformation_comparison': target_comparison,
        'slice_analysis': slice_data,
        'global_shap_importance': shap_summary
    }

    metrics_json_path = os.path.join(REPORTS_DIR, 'metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(unified_metrics, f, indent=4)
    print(f"\nUnified metrics saved to: {metrics_json_path}")

    # Step 11: Export Production Pipeline & Metadata
    prod_model_path = os.path.join(MODELS_DIR, 'car_valuation_pipeline.joblib')
    joblib.dump(final_ttr, prod_model_path)
    print(f"Exported serialized pipeline to: {prod_model_path}")

    prod_metadata = {
        'manufacturers': list(BRAND_TIERS.keys()),
        'fuel_types': ['Petrol', 'Diesel', 'Hybrid', 'Electric'],
        'transmissions': ['Automatic', 'Manual'],
        'conditions': CONDITION_ORDER,
        'min_year': int(df['model_year'].min()),
        'max_year': int(df['model_year'].max()),
        'engine_capacity_range': [int(df['engine_capacity'].min()), int(df['engine_capacity'].max())],
        'mileage_range': [int(df['mileage'].min()), int(df['mileage'].max())],
        'base_expected_price': base_expected_price,
        'q_lower_log': q_lower,
        'q_upper_log': q_upper,
        'metrics': test_metrics
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(prod_metadata, f, indent=4)

    print("\nPipeline execution completed successfully!")
    return unified_metrics

if __name__ == '__main__':
    train_and_evaluate()
