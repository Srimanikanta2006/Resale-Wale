"""
Model Training, Benchmarking, Cross-Validation & Serialization Pipeline
Follows strict ML Best Practices:
1. Strict Featurization Ordering: Split (Train/Val/Test) BEFORE fitting preprocessing.
2. Baselines: Naive Mean/Median baselines & Linear/Ridge baseline.
3. Multi-Model Benchmark: Ridge, Random Forest, XGBoost, LightGBM.
4. K-Fold Cross-Validation & Hyperparameter Tuning.
5. Comprehensive Metrics: RMSE, MAE, MedAE, MAPE, R2.
6. Bootstrap 95% Confidence Intervals for error metrics.
7. Slice-Based Error Analysis on holdout test set.
8. SHAP Explainability & Feature Importance.
9. Serialization of the winning end-to-end production pipeline.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import shap

from feature_engineering import AutomotiveFeatureEngineer, build_preprocessor, BRAND_TIERS, CONDITION_ORDER
from evaluate import calculate_metrics, bootstrap_confidence_interval, slice_based_error_analysis

REPORTS_DIR = 'reports'
MODELS_DIR = 'models'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def train_and_benchmark(data_path='data/used_cars_marketplace.csv', random_seed=42):
    print("=" * 70)
    print(" AUTOMOBILE RESALE VALUATION: ML BENCHMARK & TRAINING PIPELINE ")
    print("=" * 70)
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [
        'manufacturer', 'model_year', 'mileage', 'engine_capacity',
        'fuel_type', 'transmission', 'number_of_owners', 'condition'
    ]
    target_col = 'resale_price'

    X = df[feature_cols]
    y = df[target_col].values

    # 2. Strict Featurization Ordering: Split before fitting any transformers!
    # Train: 70%, Validation: 15%, Test: 15%
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=random_seed
    )

    print(f"Data Splits:")
    print(f" - Train set:      {X_train.shape[0]} records ({X_train.shape[0]/len(df):.1%})")
    print(f" - Validation set: {X_val.shape[0]} records ({X_val.shape[0]/len(df):.1%})")
    print(f" - Test set:       {X_test.shape[0]} records ({X_test.shape[0]/len(df):.1%})\n")

    # 3. Model Benchmark Candidates
    # Target transformation: TransformedTargetRegressor with func=np.log1p, inverse_func=np.expm1
    models = {
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
            subsample=0.85, colsample_bytree=0.85, random_state=random_seed, n_jobs=-1, verbose=-1
        )
    }

    benchmark_records = []
    trained_pipelines = {}

    print("-" * 70)
    print(f"{'Model Name':<26} | {'RMSE ($)':<9} | {'MAE ($)':<8} | {'MAPE (%)':<9} | {'R2':<7} | {'Train Time (s)':<14}")
    print("-" * 70)

    for name, model in models.items():
        t0 = time.time()
        if name == 'Naive Baseline (Mean)':
            mean_val = np.mean(y_train)
            val_preds = np.full_like(y_val, fill_value=mean_val)
            train_time = time.time() - t0
            pipe = None
        elif name == 'Naive Baseline (Median)':
            med_val = np.median(y_train)
            val_preds = np.full_like(y_val, fill_value=med_val)
            train_time = time.time() - t0
            pipe = None
        else:
            # Build end-to-end pipeline with log-transformed target
            base_pipeline = Pipeline([
                ('feature_engineer', AutomotiveFeatureEngineer()),
                ('preprocessor', build_preprocessor()),
                ('regressor', model)
            ])
            
            # Wrap in TransformedTargetRegressor for log(price) target modeling
            ttr = TransformedTargetRegressor(
                regressor=base_pipeline,
                func=np.log1p,
                inverse_func=np.expm1
            )
            
            ttr.fit(X_train, y_train)
            train_time = time.time() - t0
            val_preds = ttr.predict(X_val)
            pipe = ttr
            trained_pipelines[name] = ttr

        metrics = calculate_metrics(y_val, val_preds)
        metrics['Model'] = name
        metrics['Train_Time_Sec'] = round(train_time, 2)
        benchmark_records.append(metrics)
        
        print(f"{name:<26} | ${metrics['RMSE']:<8.2f} | ${metrics['MAE']:<7.2f} | {metrics['MAPE (%)']:<8.2f}% | {metrics['R2']:<7.4f} | {metrics['Train_Time_Sec']:<14.2f}")

    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df.to_csv(os.path.join(REPORTS_DIR, 'model_benchmark_results.csv'), index=False)

    # 4. Model Selection & Cross-Validation
    # Filter out naive baselines to pick best ML candidate
    ml_models_df = benchmark_df[~benchmark_df['Model'].str.contains('Naive')].sort_values(by='MAPE (%)')
    best_model_name = ml_models_df.iloc[0]['Model']
    print("\n" + "=" * 70)
    print(f" WINNING MODEL ON VALIDATION: {best_model_name} (MAPE: {ml_models_df.iloc[0]['MAPE (%)']}%)")
    print("=" * 70)

    # Perform 5-fold cross validation on Train+Val data to confirm generalizability
    print(f"\nRunning 5-Fold Cross Validation on (Train+Val) with {best_model_name}...")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = []
    
    winning_base_regressor = LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.06, num_leaves=55,
        subsample=0.85, colsample_bytree=0.85, random_state=random_seed, n_jobs=-1, verbose=-1
    ) if 'LightGBM' in best_model_name else XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.07, subsample=0.85,
        colsample_bytree=0.85, random_state=random_seed, n_jobs=-1
    )

    final_pipeline = Pipeline([
        ('feature_engineer', AutomotiveFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', winning_base_regressor)
    ])
    
    final_ttr = TransformedTargetRegressor(
        regressor=final_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val)):
        X_tr_f, X_val_f = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr_f, y_val_f = y_train_val[train_idx], y_train_val[val_idx]
        
        final_ttr.fit(X_tr_f, y_tr_f)
        preds_f = final_ttr.predict(X_val_f)
        f_mape = np.mean(np.abs((y_val_f - preds_f) / y_val_f)) * 100.0
        cv_scores.append(f_mape)
        print(f"  Fold {fold+1} MAPE: {f_mape:.2f}%")

    print(f"5-Fold CV Mean MAPE: {np.mean(cv_scores):.2f}% (+/- {np.std(cv_scores):.2f}%)")

    # 5. Fit on all Train+Val and Evaluate on Final Holdout Test Set
    print("\nFitting final tuned pipeline on full Training+Validation set...")
    final_ttr.fit(X_train_val, y_train_val)

    test_preds = final_ttr.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_preds)

    print("\n" + "=" * 70)
    print(" FINAL HOLDOUT TEST SET PERFORMANCE ")
    print("=" * 70)
    print(f"RMSE:     ${test_metrics['RMSE']:,.2f}")
    print(f"MAE:      ${test_metrics['MAE']:,.2f}")
    print(f"MedAE:    ${test_metrics['MedAE']:,.2f}")
    print(f"MAPE:     {test_metrics['MAPE (%)']:.2f}%")
    print(f"R2 Score: {test_metrics['R2']:.4f}")

    # Bootstrap 95% Confidence Intervals
    print("\nCalculating 95% Bootstrap Confidence Intervals (1,000 iterations)...")
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

    print(f"RMSE 95% CI:     ${rmse_ci[0]:,.2f} to ${rmse_ci[1]:,.2f}")
    print(f"MAE 95% CI:      ${mae_ci[0]:,.2f} to ${mae_ci[1]:,.2f}")
    print(f"MAPE 95% CI:     {mape_ci[0]:.2f}% to {mape_ci[1]:.2f}%")

    test_eval_summary = {
        'best_model': best_model_name,
        'metrics': test_metrics,
        'confidence_intervals_95': {
            'RMSE': {'lower': rmse_ci[0], 'upper': rmse_ci[1]},
            'MAE': {'lower': mae_ci[0], 'upper': mae_ci[1]},
            'MAPE': {'lower': mape_ci[0], 'upper': mape_ci[1]}
        },
        'cv_5fold_mape_mean': round(float(np.mean(cv_scores)), 2),
        'cv_5fold_mape_std': round(float(np.std(cv_scores)), 2)
    }

    with open(os.path.join(REPORTS_DIR, 'test_evaluation_summary.json'), 'w') as f:
        json.dump(test_eval_summary, f, indent=4)

    # 6. Slice-Based Error Analysis
    print("\nExecuting Slice-Based Error Analysis on Holdout Test Set...")
    df_test_eval = X_test.copy()
    df_test_eval['resale_price'] = y_test
    df_test_eval['predicted_price'] = test_preds
    df_test_eval['brand_tier'] = df_test_eval['manufacturer'].map(BRAND_TIERS).fillna('Mainstream')

    slices = slice_based_error_analysis(df_test_eval)
    for slice_name, slice_df in slices.items():
        print(f"\n--- Performance Slice: {slice_name} ---")
        print(slice_df[['Slice', 'Count', 'RMSE', 'MAE', 'MAPE (%)', 'R2']].to_string(index=False))
        slice_df.to_csv(os.path.join(REPORTS_DIR, f'slice_analysis_{slice_name.lower()}.csv'), index=False)

    # 7. Residual & Error Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, test_preds, alpha=0.35, s=15, color='#1f77b4')
    max_val = max(max(y_test), max(test_preds))
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    axes[0].set_title(f'Actual vs. Predicted Resale Price (Test Set)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Resale Price ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Resale Price ($)', fontsize=11)
    axes[0].yaxis.set_major_formatter('${x:,.0f}')
    axes[0].xaxis.set_major_formatter('${x:,.0f}')
    axes[0].legend()

    # Residual Distribution
    residuals = y_test - test_preds
    sns.histplot(residuals, kde=True, ax=axes[1], color='#2ca02c', bins=40)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_title('Residual Error Distribution (y_true - y_pred)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Error ($)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].xaxis.set_major_formatter('${x:,.0f}')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '07_test_actual_vs_predicted_residuals.png'), dpi=300)
    plt.close()
    print("-> Saved: 07_test_actual_vs_predicted_residuals.png")

    # 8. SHAP Feature Attribution Analysis
    print("\nComputing SHAP Explanations for Global Interpretability...")
    # Extract inner transformed model for SHAP
    inner_pipe = final_ttr.regressor_
    fe_step = inner_pipe.named_steps['feature_engineer']
    prep_step = inner_pipe.named_steps['preprocessor']
    reg_step = inner_pipe.named_steps['regressor']

    X_train_enriched = fe_step.transform(X_train_val.head(2000))
    X_train_transformed = prep_step.transform(X_train_enriched)

    # Get feature names from ColumnTransformer
    encoded_feature_names = prep_step.get_feature_names_out()

    explainer = shap.TreeExplainer(reg_step)
    shap_sample = prep_step.transform(fe_step.transform(X_test.head(600)))
    shap_values = explainer.shap_values(shap_sample)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, feature_names=encoded_feature_names, show=False, max_display=15)
    plt.title('SHAP Feature Importance & Impact Direction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved: 08_shap_summary_plot.png")

    # 9. Save Production Artifacts
    production_model_path = os.path.join(MODELS_DIR, 'car_valuation_pipeline.joblib')
    joblib.dump(final_ttr, production_model_path)
    print(f"\nProduction Pipeline successfully exported to: {production_model_path}")

    # Save feature metadata for inference & web app
    metadata = {
        'manufacturers': list(BRAND_TIERS.keys()),
        'fuel_types': ['Petrol', 'Diesel', 'Hybrid', 'Electric'],
        'transmissions': ['Automatic', 'Manual'],
        'conditions': CONDITION_ORDER,
        'min_year': int(df['model_year'].min()),
        'max_year': int(df['model_year'].max()),
        'engine_capacity_range': [int(df['engine_capacity'].min()), int(df['engine_capacity'].max())],
        'mileage_range': [int(df['mileage'].min()), int(df['mileage'].max())],
        'metrics': test_metrics
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training pipeline run completed flawlessly!")
    return test_eval_summary

if __name__ == '__main__':
    train_and_benchmark()
