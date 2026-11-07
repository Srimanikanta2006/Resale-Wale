# 🚗 Resale Wale - Used Car Resale Price Valuation Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM_Regressor-brightgreen)](https://lightgbm.readthedocs.io/)
[![R2 Score](https://img.shields.io/badge/Holdout_R²-0.9640-blue)](models/)
[![MAPE](https://img.shields.io/badge/Test_MAPE-8.03%25-success)](reports/)
[![FastAPI](https://img.shields.io/badge/Production_Backend-FastAPI-009688?logo=fastapi&logoColor=white)](http://localhost:8000)
[![Interface](https://img.shields.io/badge/UI-Interactive_HTML5_Canvas-111111)](http://localhost:8000)
[![Pytest](https://img.shields.io/badge/Tests-Passing-brightgreen?logo=pytest)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**An Enterprise-Grade, Machine Learning Marketplace Valuation System with Domain Feature Engineering, Explainable AI, and Interactive Pricing Intelligence.**

[Live Web Application (Port 8000)](http://localhost:8000) • [Research Notebook](notebooks/used_car_valuation_study.ipynb)

</div>

---

## 📌 1. Project Overview & Problem Statement
Used automobile marketplaces (such as *Kelley Blue Book*, *Edmunds*, *Cars24*, *Carvana*, *Autotrader*, *CarDekho*) require high-precision automated pricing engines to calculate fair resale prices, eliminate pricing friction between buyers and sellers, and protect dealer margins.

This project delivers an end-to-end Machine Learning system strictly addressing all 8 required features:
1. **Manufacturer (Brand)**: e.g., Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Volkswagen, BMW, Mercedes-Benz, Audi
2. **Model Year**: Year of vehicle manufacture (2008–2024)
3. **Mileage**: Total odometer reading (miles)
4. **Engine Capacity**: Engine displacement in cubic centimeters (cc) / Liters (with 0 cc handling for Electric vehicles)
5. **Fuel Type**: Petrol, Diesel, Hybrid, Electric
6. **Transmission**: Automatic, Manual
7. **Number of Owners**: 1st Owner, 2nd Owner, 3rd Owner, 4th+ Owner
8. **Condition**: Fair, Good, Very Good, Excellent

**Target Variable**: Resale Market Value (`resale_price` in USD).

---

## 🏗️ 2. Architecture & ML Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESALE WALE ML PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  [Data Ingestion] ────► 25,000 Realistic Market Transactions (data/)
          │
          ▼
  [Strict Ordering] ───► Train (70%) / Validation (15%) / Test (15%) Holdout
          │
          ▼
  [Feature Engineering] ──► Vehicle Age, Annual Mileage, Mileage Deviation,
                             Brand Prestige Tier, Engine Displacement Tier,
                             Holistic Wear & Tear Index
          │
          ▼
  [Preprocessing] ─────► StandardScaler (Num), OrdinalEncoder (Condition),
                             OneHotEncoder (Categoricals)
          │
          ▼
  [Log-Target Regressor] ► log1p(resale_price) Variance Stabilization
          │
          ▼
  [Model Benchmark] ───► Naive Mean/Median, Ridge, Random Forest, XGBoost, LightGBM
          │
          ▼
  [Rigorous Validation] ► 5-Fold Cross Validation + 1,000 Bootstrap 95% CIs
          │
          ▼
  [Interpretability] ──► Global & Local TreeExplainer SHAP Values
          │
          ▼
  [Production Serving] ─► Serialized Pipeline (models/) + Web Application (static/)
```

### Key Engineering Decisions (ML Best Practices):
1. **Strict Featurization Ordering**: Datasets were cleanly partitioned into Train, Validation, and Test sets *prior* to fitting any scalers or encoders, eliminating data leakage.
2. **Log-Transformed Target Modeling**: Automobile prices exhibit right-skew and heteroscedasticity. By training on $\log(1 + y)$ and transforming back with $\exp(y) - 1$, predictions are guaranteed positive and optimization directly minimizes **Mean Absolute Percentage Error (MAPE)**.
3. **Domain Feature Engineering**:
   - `vehicle_age = 2024 - model_year`
   - `annual_mileage = mileage / max(vehicle_age, 0.5)`
   - `mileage_vs_expected = mileage - (vehicle_age * 12,000)` (identifies abused vs. garage-kept cars)
   - `brand_tier = Economy | Mainstream | Luxury`
   - `engine_tier = SubCompact (<1.4L) | Compact_Midsize (1.4-2.2L) | Large_V6 (2.2-3.5L) | Performance_V8 (>3.5L) | Zero_EV (0 cc)`
   - `wear_index = (age * 0.40) + (owners * 0.70) + (annual_mileage / 10,000 * 0.35)`

---

## 📊 3. Model Benchmark Results

All candidate models were trained on the training partition (17,499 records) and evaluated on identical validation partitions (3,751 records):

| Model Architecture | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Mean)** | \$11,596.61 | \$8,560.52 | 75.24% | -0.0000 | 0.00s |
| **Naive Baseline (Median)** | \$11,801.01 | \$8,367.50 | 65.22% | -0.0356 | 0.00s |
| **Ridge Regression (L2)** | \$2,811.35 | \$1,711.99 | 9.33% | 0.9412 | 0.11s |
| **Random Forest Regressor** | \$2,430.94 | \$1,567.20 | 8.74% | 0.9561 | 1.43s |
| **XGBoost Regressor** | \$2,219.99 | \$1,438.34 | 7.96% | 0.9634 | 0.53s |
| **LightGBM Regressor (Winner)** | **\$2,204.68** | **\$1,429.57** | **7.95%** | **0.9639** | **0.47s** |

### 5-Fold Cross Validation (Stability Check):
- Fold 1: 8.00% MAPE
- Fold 2: 8.26% MAPE
- Fold 3: 7.92% MAPE
- Fold 4: 8.00% MAPE
- Fold 5: 7.96% MAPE
- **Mean CV MAPE**: **8.03% ($\pm 0.12\%$)**

---

## 🎯 4. Final Holdout Test Set Performance & Statistical Rigor

The final winning pipeline was refit on Train+Val and evaluated on the held-out **Test Set (3,750 unseen vehicles)**:

| Metric | Point Estimate | 95% Bootstrap Confidence Interval (1,000 runs) |
| :--- | :---: | :---: |
| **$R^2$ Score** | **0.9640** | [0.9602, 0.9674] |
| **MAPE (%)** | **8.03%** | **[7.84%, 8.22%]** |
| **MAE ($)** | **\$1,419.02** | **[\$1,373.45, \$1,470.31]** |
| **RMSE ($)** | **\$2,137.48** | **[\$2,045.03, \$2,239.42]** |
| **Median AE ($)**| **\$919.74** | -- |

---

## 🔍 5. Slice-Based Error Analysis

To ensure no hidden algorithmic bias across sub-populations, slice-based evaluations were conducted:

### Vehicle Age Slices:
| Age Bracket | Sample Count | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< 3 Years (Newer)** | 926 | \$3,260.46 | \$2,392.35 | 8.01% | 0.9265 |
| **3–7 Years (Mid-Age)** | 1,446 | \$2,044.02 | \$1,485.82 | 7.97% | 0.9170 |
| **7–12 Years (Mature)** | 1,054 | \$1,034.23 | \$769.86 | 7.91% | 0.9259 |
| **> 12 Years (Older)** | 324 | \$609.32 | \$450.84 | 8.71% | 0.8405 |

### Brand Market Tier Slices:
| Brand Tier | Sample Count | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Economy (Nissan, Hyundai)** | 777 | \$1,301.21 | \$950.09 | 7.77% | 0.9573 |
| **Mainstream (Toyota, Honda, Ford, Chevy, VW)** | 2,330 | \$1,853.59 | \$1,309.34 | 8.00% | 0.9494 |
| **Luxury (BMW, Mercedes-Benz, Audi)** | 643 | \$3,485.60 | \$2,383.11 | 8.42% | 0.9585 |

### Condition Grade Slices:
| Condition Grade | Sample Count | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fair** | 716 | \$689.68 | \$527.84 | 8.50% | 0.9021 |
| **Good** | 1,168 | \$1,435.33 | \$1,037.17 | 7.81% | 0.9221 |
| **Very Good** | 1,369 | \$2,437.75 | \$1,780.95 | 7.98% | 0.9240 |
| **Excellent** | 497 | \$3,546.40 | \$2,603.32 | 7.99% | 0.9241 |

---

## 💡 6. SHAP Explainability & Global Attribution
SHAP (TreeExplainer) attribution confirms strong alignment with automotive economic fundamentals:
- **Vehicle Age & Model Year**: The dominant drivers of value decay across all segments.
- **Brand Prestige Tier**: Strong positive shift for European luxury brands (BMW, Mercedes, Audi) and high retention for Toyota/Honda.
- **Condition Grading**: "Excellent" adds a noticeable positive shift, while "Fair" applies a steep negative penalty for required mechanical reconditioning.
- **Mileage vs. Expected**: Vehicles driven significantly less than 12,000 miles/year earn preservation premiums; excess mileage triggers progressive penalties.

---

## 🖥️ 7. Web Application & Interactive Platform

The platform is served by a high-performance FastAPI server (`src/api_server.py`) delivering a single-file, light-themed responsive web application (`static/index.html`):
- **Kindred Dot-Matrix Canvas Hero**:
  - Live `<canvas>` rendering 2 Creation-of-Adam hands reaching toward each other in a black dot-matrix / LED halftone scanline pattern.
  - Slices a 2x2 Higgsfield photographic luminance sheet into pre-downscaled tiles, drawing into offscreen field buffers `fA` (Pose A: pointing pair) and `fB` (Pose B: open pair).
  - 20-second counter-phase continuous drift, breathing scale, roll rotation, and periodic glitch-dissolve pose switches with row streaks and sparkle fronts.
  - Center quiet zone preserving title legibility, rainbow spectrum underglow CTA pill, and inline SVG automotive brand logos.
- **Unified Resale Wale Appraisal Workbench**:
  1. **Valuation Estimator**: Real-time pricing calculator for all 8 parameters with live confidence intervals, Trade-In, Private Party, and Dealer Retail bands, plus 5-year depreciation schedules.
  2. **Explainable AI & Diagnostics**: Interactive SHAP waterfall chart, top 5 value drivers, and counterfactual simulation sandbox ("What if I drove 15k fewer miles or detailed to Excellent?").
  3. **Marketplace Intelligence & Comps**: Live interactive scatter plot of comps, regression line, brand tier filtering, and market comps table directly querying the 25,000 dataset.
  4. **MLOps & Model Benchmarks**: Live model telemetry, latency benchmarks, full 6-model comparison table, and slice-based error analysis across age, brand tier, and condition.

---

## 📂 8. Repository Structure

```
Resale Wale/
├── .venv/                         # Project virtual environment
├── data/
│   └── used_cars_marketplace.csv  # 25,000-sample curated marketplace dataset
├── models/
│   ├── car_valuation_pipeline.joblib # Production serialized ML pipeline
│   └── model_metadata.json        # Pipeline feature schema & metrics metadata
├── notebooks/
│   └── used_car_valuation_study.ipynb # Publication-quality narrative notebook
├── reports/
│   ├── figures/                   # 8 High-resolution visualizations & SHAP plots
│   ├── model_benchmark_results.csv# Benchmark comparison across models
│   ├── slice_analysis_*.csv       # Slice-based evaluation breakdowns
│   └── test_evaluation_summary.json# Test metrics & 95% bootstrap CIs
├── static/
│   ├── index.html                 # Flagship Resale Wale Web App (Single-file)
│   └── hands.webp                 # Cached Higgsfield 2x2 hands photo sheet
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py        # Realistic automotive benchmark generator
│   ├── feature_engineering.py     # Domain transformers & ColumnTransformer
│   ├── train.py                   # Multi-model benchmarking, CV & SHAP
│   ├── evaluate.py                # Statistical testing, bootstrapping & slices
│   ├── predict.py                 # Production inference API & depreciation forecaster
│   └── api_server.py              # FastAPI production server & static dispatcher
├── tests/
│   └── test_pipeline.py           # Pytest suite for preprocessing, logic & inference
├── requirements.txt               # Pinned Python package dependencies
├── LICENSE                        # MIT License
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 9. Quickstart Guide

### 1. Activate Environment & Install Dependencies
```bash
# On Windows
.venv\Scripts\activate

# Dependencies are tracked in requirements.txt
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
```

### 3. Generate Data, Run EDA & Train Models
```bash
# Generate 25,000 record dataset
python src/generate_dataset.py

# Run Exploratory Data Analysis
python src/eda_analysis.py

# Benchmark all models & export production pipeline
python src/train.py
```

### 4. Launch Production Web App
```bash
python src/api_server.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

