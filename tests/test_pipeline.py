"""
Unit & Integration Tests for Automobile Resale Marketplace ML System
Tests data generation, feature engineering, pipeline inference, and business logic.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_engineering import AutomotiveFeatureEngineer, build_preprocessor, BRAND_TIERS, CONDITION_ORDER
from predict import CarValuationEngine

def test_feature_engineering_columns():
    fe = AutomotiveFeatureEngineer(current_year=2024)
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
    
    assert df_out['vehicle_age'].iloc[0] == 6
    assert df_out['annual_mileage'].iloc[0] == 10000.0
    assert df_out['brand_tier'].iloc[0] == 'Mainstream'
    assert df_out['engine_tier'].iloc[0] == 'Large_V6'

def test_model_inference_pipeline():
    engine = CarValuationEngine()
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
    assert result['fair_resale_price'] > 5000.0
    assert result['trade_in_value'] < result['fair_resale_price']
    assert result['dealer_retail_value'] > result['fair_resale_price']
    
    # 95% Confidence Interval check
    low, high = result['prediction_interval_95']
    assert low < result['fair_resale_price'] < high

def test_condition_price_monotonicity():
    engine = CarValuationEngine()
    fair_car = engine.predict_car_value(
        manufacturer='Ford', model_year=2017, mileage=75000, engine_capacity=1999,
        fuel_type='Petrol', transmission='Automatic', number_of_owners=2, condition='Fair'
    )
    good_car = engine.predict_car_value(
        manufacturer='Ford', model_year=2017, mileage=75000, engine_capacity=1999,
        fuel_type='Petrol', transmission='Automatic', number_of_owners=2, condition='Good'
    )
    exc_car = engine.predict_car_value(
        manufacturer='Ford', model_year=2017, mileage=75000, engine_capacity=1999,
        fuel_type='Petrol', transmission='Automatic', number_of_owners=2, condition='Excellent'
    )
    
    assert fair_car['fair_resale_price'] < good_car['fair_resale_price'] < exc_car['fair_resale_price']

def test_depreciation_forecast():
    engine = CarValuationEngine()
    forecast = engine.forecast_depreciation(
        manufacturer='BMW', model_year=2021, mileage=30000, engine_capacity=2998,
        fuel_type='Petrol', transmission='Automatic', number_of_owners=1, condition='Excellent',
        years_ahead=3
    )
    assert len(forecast) == 4 # Year 0, 1, 2, 3
    assert forecast['projected_price'].iloc[0] > forecast['projected_price'].iloc[-1]
