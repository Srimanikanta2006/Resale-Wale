# 🚗 Resale Wale - Used Car Resale Price Valuation Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-Monotonic_LightGBM-brightgreen)](https://lightgbm.readthedocs.io/)
[![R2 Score](https://img.shields.io/badge/Holdout_R%C2%B2-0.9629-blue)](reports/metrics.json)
[![MAPE](https://img.shields.io/badge/Holdout_MAPE-8.14%25-success)](reports/metrics.json)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_REST-009688?logo=fastapi&logoColor=white)](http://localhost:8000)
[![Pytest](https://img.shields.io/badge/Tests-8%20Passed-brightgreen?logo=pytest)](tests/)
[![Single Source](https://img.shields.io/badge/Truth-reports%2Fmetrics.json-purple)](reports/metrics.json)

**A Clean 8-Feature Automobile Valuation Engine with Monotonic Gradient Boosted Trees, Asymmetric Prediction Intervals, and TreeSHAP Explainability.**

[Live Web Interface (Port 8000)](http://localhost:8000) • [API Documentation](http://localhost:8000/docs) • [Unified Metrics](reports/metrics.json)

</div>

---

## 📌 1. Operational Scope & Problem Statement
Used automobile marketplaces require transparent, data-driven pricing engines to calculate fair resale prices. This system is designed as a **clean 8-input → price tool**, honest about what its 25,000 transaction records can and cannot cover:

### Valid Operational Domain:
- **10 Manufacturers**: Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Volkswagen, Audi, BMW, Mercedes-Benz
- **Model Years**: 2008 – 2024 (valued as of **2024 calendar market baseline**, unindexed for subsequent macro inflation)
- **Odometer Range**: 500 – 300,000 miles
- **8 Core Vehicular Features (strictly respecting the problem statement)**:
  1. `manufacturer` (Make)
  2. `model_year` (2008–2024)
  3. `mileage` (Odometer in miles)
  4. `engine_capacity` (Displacement in cc; strictly 0 cc for Electric vehicles)
  5. `fuel_type` (Petrol, Diesel, Hybrid, Electric)
  6. `transmission` (Automatic, Manual; strictly Automatic for Electric vehicles)
  7. `number_of_owners` (1, 2, 3, 4+ Owners)
  8. `condition` (Fair, Good, Very Good, Excellent)
- **Target**: `resale_price` in USD ($).

*Note on Boundaries:* Specific vehicle trim packages, exterior colors, and regional taxes are omitted because they are not present in the 8-feature problem statement.

---

## 🏗️ 2. Methodology & ML Best Practices

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESALE WALE ML ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             [Data Ingestion] ────► 25,000 Marketplace Records (data/)
                                      │
                                      ▼
             [Strict Split]   ────► Train (70%) / Val (15%) / Test (15%) Holdout
                                      │
                                      ▼
             [Domain Features]───► vehicle_age, annual_mileage, mileage_vs_expected,
                                    brand_tier, engine_tier, wear_index
                                      │
                                      ▼
             [Preprocessing]  ───► StandardScaler, OrdinalEncoder, OneHotEncoder
                                      │
                                      ▼
             [Monotonic Trees]───► LightGBM with Monotone Constraints
                                      │
                                      ▼
             [Target Modeling]───► TransformedTargetRegressor(log1p, expm1)
                                      │
                                      ▼
             [Residual Quantiles]► Asymmetric 95% Prediction Interval in Log Space
                                      │
                                      ▼
             [Explainability] ───► TreeSHAP (Base Price + Drivers = Total)
```

### Core Engineering Decisions:
1. **Strict Featurization Ordering**: Data was cleanly partitioned into Train (17,499), Validation (3,751), and Holdout Test (3,750) sets **before** fitting any scalers or encoders, eliminating all data leakage.
2. **Log-Transformed Target**: Automobile prices are right-skewed and heteroscedastic. By training on $\log(1 + y)$ and inverse-transforming with $\exp(\hat{z}) - 1$, predictions are guaranteed positive and percentage error (MAPE) is minimized.
3. **Monotonic Constraints**: LightGBM enforces strict economic monotonicity:
   - More mileage never raises price ($\le 0$).
   - Older vehicle age never raises price ($\le 0$).
   - More previous owners never raises price ($\le 0$).
   - Superior condition grade never lowers price ($\ge 0$).
4. **Asymmetric 95% Prediction Intervals**: Derived from validation set empirical residual quantiles in log space ($q_{0.025} = -0.1965$, $q_{0.975} = +0.1848$). Exponentiating to dollar space naturally produces an asymmetric prediction interval reflecting price dispersion, achieving **93.15% coverage** on unseen test data.

---

## 📊 3. Performance Metrics (Single Source of Truth: `reports/metrics.json`)

### Candidate Benchmark Comparison (Trained on 70% Train, Evaluated on 15% Validation)
| Model Architecture | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Mean)** | $11,596.61 | $8,560.52 | 75.24% | -0.0000 | Rejected Baseline |
| **Naive Baseline (Median)** | $11,801.01 | $8,367.50 | 65.22% | -0.0356 | Rejected Baseline |
| **Ridge Regression (L2)** | $2,811.35 | $1,711.99 | 9.33% | 0.9412 | Linear Baseline |
| **Random Forest (100 Trees)** | $2,430.94 | $1,567.20 | 8.74% | 0.9561 | Benchmark Candidate |
| **XGBoost Regressor** | $2,219.99 | $1,438.34 | 7.96% | 0.9634 | Benchmark Candidate |
| **LightGBM Regressor (Winner)** | **$2,254.93** | **$1,472.60** | **8.19%** | **0.9622** | **Production Winner** |

### Final Champion Model (Refit on 21,250 records, Evaluated on 3,750 Held-Out Test Set)
- **Holdout Test MAPE**: **8.14%** (95% Bootstrap CI: **[7.95%, 8.34%]**)
- **Holdout Test $R^2$ Score**: **0.9629**
- **Mean Absolute Error (MAE)**: **$1,437.13** (Median AE: **$938.12**)
- **Root Mean Squared Error (RMSE)**: **$2,167.80**
- **5-Fold Cross Validation Mean MAPE**: **8.22% ($\pm 0.15\%$)**
- **Inference Latency**: **14.8 ms** (Real-time REST API)

---

## 🧪 4. Sanity Test Suite
Automated pytest tests in `tests/test_pipeline.py` enforce behavioral consistency:

| Sanity Check | Test Condition | Status |
| :--- | :--- | :---: |
| **Mileage Monotonicity** | As mileage increases from 15k to 140k mi, price strictly decreases | **PASSED** |
| **Age Monotonicity** | As model year decreases (age increases), price strictly decreases | **PASSED** |
| **Owner Monotonicity** | As owner count increases from 1 to 4, price strictly decreases | **PASSED** |
| **Condition Monotonicity** | Fair < Good < Very Good < Excellent | **PASSED** |
| **EV Constraints** | Electric fuel type forces 0 cc displacement and Automatic transmission | **PASSED** |
| **Non-Negative Guarantee** | Even at 320,000 miles, 2008 year, Fair condition, price remains > 0 | **PASSED** |
| **Prediction Interval** | 95% log-space prediction interval covers point estimate | **PASSED** |

Run tests anytime:
```bash
pytest tests/
```

---

## 🚀 5. How to Run the Project

### Prerequisites
- Python 3.10+ (Recommended Python 3.11)
- Standard dependencies in virtual environment

```bash
# 1. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI Production Server
python src/api_server.py
```
- Web Application: [http://localhost:8000](http://localhost:8000)
- Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Reproduce Model Training & Metrics from Scratch
```bash
python src/train.py
```
*Outputs:*
- `reports/metrics.json` (Single source of truth)
- `reports/model_benchmark_results.csv`
- `reports/slice_analysis_*.csv`
- `models/car_valuation_pipeline.joblib`

---

## 📜 6. Repository Structure
```
├── data/
│   └── used_cars_marketplace.csv     # 25,000 transaction records
├── models/
│   ├── car_valuation_pipeline.joblib # Serialized production pipeline
│   └── model_metadata.json           # Model configuration metadata
├── reports/
│   ├── metrics.json                  # Single source of truth for metrics
│   ├── model_benchmark_results.csv   # Validation partition benchmark
│   └── slice_analysis_*.csv          # Sliced error analysis (Age, Brand, Condition)
├── src/
│   ├── api_server.py                 # FastAPI microservice
│   ├── feature_engineering.py        # Domain transformers & preprocessor
│   ├── predict.py                    # Valuation engine & TreeSHAP attributions
│   └── train.py                      # Monotonic training & governance pipeline
├── static/
│   └── index.html                    # Responsive web terminal (8 inputs, SHAP, comps)
├── tests/
│   └── test_pipeline.py              # 8 automated unit & sanity monotonicity tests
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Project documentation
```
