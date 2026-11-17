"""
Unit & Integration Sanity Test Suite for Resale Wale ML Valuation System
Tests:
1. Feature engineering domain transformations.
2. Pipeline inference & asymmetric log-space prediction intervals.
3. Strict monotonicity verification:
   - More mileage never raises price.
   - Older vehicle age (older model year) never raises price.
   - More previous owners never raises price.
   - Better physical condition never lowers price.
4. Non-negative price guarantees.
5. EV domain constraint enforcement (Electric -> 0 cc, Automatic).
6. 5-Year scenario simulation.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_engineering import AutomotiveFeatureEngineer, BRAND_TIERS, CONDITION_ORDER
from predict import CarValuationEngine

@pytest.fixture(scope="module")
def engine():
    return CarValuationEngine()

def test_feature_engineering_columns():
    fe = AutomotiveFeatureEngineer(current_year=2026)
    df_raw = pd.DataFrame([{
        'manufacturer': 'Toyota',
        'model_year': 2018,
        'mileage': 60000,
        'engine_capacity': 2494,
        'fuel_type': 'Petrol',
        'transmission': 'Automatic',
        'number_of_owners': 1,
        'condition': 'Very Good'
    }])
    df_out = fe.transform(df_raw)
    
    expected_cols = [
        'vehicle_age', 'annual_mileage', 'mileage_vs_expected',
        'brand_tier', 'engine_tier', 'wear_index'
    ]
    for col in expected_cols:
        assert col in df_out.columns, f"Missing engineered column {col}"
    
    assert df_out['vehicle_age'].iloc[0] == 8
    assert df_out['annual_mileage'].iloc[0] == 7500.0
    assert df_out['brand_tier'].iloc[0] == 'Mainstream'
    assert df_out['engine_tier'].iloc[0] == 'Large_V6'

def test_model_inference_and_prediction_interval(engine):
    result = engine.predict_car_value(
        manufacturer='Honda',
        model_year=2020,
        mileage=40000,
        engine_capacity=1498,
        fuel_type='Petrol',
        transmission='Automatic',
        number_of_owners=1,
        condition='Excellent'
    )
    
    assert 'fair_resale_price' in result
    assert result['fair_resale_price'] > 1800.0
    assert result['trade_in_value'] < result['fair_resale_price']
    assert result['dealer_retail_value'] > result['fair_resale_price']
    
    # Check asymmetric 95% prediction interval
    low, high = result['prediction_interval_95']
    assert low < result['fair_resale_price'] < high

def test_mileage_monotonicity(engine):
    """
    Sanity test: Higher mileage must never increase predicted resale price.
    """
    mileages = [15000, 35000, 60000, 95000, 140000]
    prices = []
    for m in mileages:
        res = engine.predict_car_value(
            manufacturer='BMW', model_year=2021, mileage=m, engine_capacity=1998,
            fuel_type='Petrol', transmission='Automatic', number_of_owners=1, condition='Excellent'
        )
        prices.append(res['fair_resale_price'])
    
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1], f"Mileage monotonicity violated at index {i}: {prices[i]} < {prices[i+1]}"

def test_age_monotonicity(engine):
    """
    Sanity test: Older manufacture year (greater vehicle age) must never increase predicted price.
    """
    years = [2023, 2021, 2018, 2014, 2010]
    prices = []
    for y in years:
        res = engine.predict_car_value(
            manufacturer='Toyota', model_year=y, mileage=50000, engine_capacity=2494,
            fuel_type='Petrol', transmission='Automatic', number_of_owners=1, condition='Good'
        )
        prices.append(res['fair_resale_price'])
    
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1], f"Age monotonicity violated at index {i}: {prices[i]} < {prices[i+1]}"

def test_owners_monotonicity(engine):
    """
    Sanity test: More previous owners must never increase predicted price.
    """
    prices = []
    for o in [1, 2, 3, 4]:
        res = engine.predict_car_value(
            manufacturer='Audi', model_year=2019, mileage=45000, engine_capacity=1998,
            fuel_type='Petrol', transmission='Automatic', number_of_owners=o, condition='Very Good'
        )
        prices.append(res['fair_resale_price'])
    
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1], f"Owner monotonicity violated at index {i}: {prices[i]} < {prices[i+1]}"

def test_condition_monotonicity(engine):
    """
    Sanity test: Better condition must never decrease predicted price.
    """
    prices = []
    for c in ['Fair', 'Good', 'Very Good', 'Excellent']:
        res = engine.predict_car_value(
            manufacturer='Ford', model_year=2018, mileage=60000, engine_capacity=2488,
            fuel_type='Petrol', transmission='Automatic', number_of_owners=2, condition=c
        )
        prices.append(res['fair_resale_price'])
    
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1], f"Condition monotonicity violated at index {i}: {prices[i]} > {prices[i+1]}"

def test_ev_constraint_enforcement(engine):
    """
    Sanity test: Electric fuel type forces 0 cc displacement and Automatic transmission.
    """
    res = engine.predict_car_value(
        manufacturer='Hyundai', model_year=2022, mileage=25000, engine_capacity=3300, # Invalid CC passed
        fuel_type='Electric', transmission='Manual', # Invalid transmission passed
        number_of_owners=1, condition='Excellent'
    )
    clean = res['clean_inputs']
    assert clean['engine_capacity'] == 0.0, "EV did not force engine capacity to 0 cc"
    assert clean['transmission'] == 'Automatic', "EV did not force transmission to Automatic"
    assert res['fair_resale_price'] > 0.0

def test_non_negative_price_guarantee(engine):
    """
    Sanity test: High mileage, old age, poor condition must never produce negative prices.
    """
    res = engine.predict_car_value(
        manufacturer='Nissan', model_year=2008, mileage=320000, engine_capacity=1498,
        fuel_type='Petrol', transmission='Manual', number_of_owners=4, condition='Fair'
    )
    assert res['fair_resale_price'] >= 1800.0
    low, high = res['prediction_interval_95']
    assert low >= 1800.0 and high >= 1800.0

def test_selling_window_counterfactuals(engine):
    """
    Sanity test: Optimal selling window counterfactuals project non-increasing resale values.
    """
    res = engine.predict_car_value(
        manufacturer='BMW', model_year=2022, mileage=30000, engine_capacity=1998,
        fuel_type='Petrol', transmission='Automatic', number_of_owners=1, condition='Excellent'
    )
    sw = res['selling_window']
    assert 'sell_now' in sw and 'in_6_months' in sw and 'in_12_months' in sw
    p_now = sw['sell_now']['projected_price']
    p_6m = sw['in_6_months']['projected_price']
    p_12m = sw['in_12_months']['projected_price']
    assert p_now >= p_6m >= p_12m, f"Selling window violated monotonicity: {p_now} -> {p_6m} -> {p_12m}"
