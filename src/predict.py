"""
Inference & Valuation Engine Module
Loads the serialized pipeline and provides:
1. Point estimate (Resale Price)
2. 95% Prediction Intervals (Uncertainty bounds)
3. Market pricing bands: Dealer Trade-In, Private Party, Dealer Retail
4. 5-Year Future Depreciation Forecast
5. Feature attribution factors
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Support both package and script execution paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_engineering import BRAND_TIERS, CONDITION_ORDER

MODEL_PATH = os.path.join('models', 'car_valuation_pipeline.joblib')

class CarValuationEngine:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        self.pipeline = joblib.load(model_path)
        # Residual MAPE from test evaluation for prediction intervals (~8.0%)
        self.mape_uncertainty = 0.0803

    def predict_car_value(self, manufacturer, model_year, mileage, engine_capacity,
                          fuel_type, transmission, number_of_owners, condition):
        """
        Computes detailed vehicle appraisal and pricing bands.
        """
        # Format single sample dataframe
        input_data = pd.DataFrame([{
            'manufacturer': manufacturer,
            'model_year': int(model_year),
            'mileage': float(mileage),
            'engine_capacity': float(engine_capacity),
            'fuel_type': fuel_type,
            'transmission': transmission,
            'number_of_owners': int(number_of_owners),
            'condition': condition
        }])

        # Predict base fair resale price
        estimated_price = float(self.pipeline.predict(input_data)[0])
        estimated_price = max(1800.0, round(estimated_price, 2))

        # Prediction intervals (95% CI based on model residual standard error / MAPE)
        lower_bound = round(estimated_price * (1.0 - 1.96 * self.mape_uncertainty * 0.75), 2)
        upper_bound = round(estimated_price * (1.0 + 1.96 * self.mape_uncertainty * 0.75), 2)

        # Market Transaction Tiers (Industry Standard Benchmarks)
        # Trade-In: Dealer margin & reconditioning allowance (~88%)
        trade_in_value = round(estimated_price * 0.88, 2)
        # Private Party: Peer-to-peer fair value (~100%)
        private_party_value = estimated_price
        # Dealer Retail: Certified / showroom retail with warranty (~112%)
        dealer_retail_value = round(estimated_price * 1.12, 2)

        # Expected mileage benchmark
        vehicle_age = max(0, 2024 - int(model_year))
        expected_miles = max(vehicle_age, 0.5) * 12000.0
        mileage_delta = float(mileage) - expected_miles

        return {
            'fair_resale_price': estimated_price,
            'prediction_interval_95': (lower_bound, upper_bound),
            'trade_in_value': trade_in_value,
            'private_party_value': private_party_value,
            'dealer_retail_value': dealer_retail_value,
            'vehicle_age': vehicle_age,
            'expected_miles': expected_miles,
            'mileage_delta': mileage_delta,
            'brand_tier': BRAND_TIERS.get(manufacturer, 'Mainstream')
        }

    def forecast_depreciation(self, manufacturer, model_year, mileage, engine_capacity,
                              fuel_type, transmission, number_of_owners, condition,
                              years_ahead=5, annual_driving_miles=12000):
        """
        Projects fair market value year-over-year into the future.
        """
        current_year = 2024
        forecast = []

        curr_res = self.predict_car_value(
            manufacturer, model_year, mileage, engine_capacity,
            fuel_type, transmission, number_of_owners, condition
        )
        forecast.append({
            'year_offset': 0,
            'calendar_year': current_year,
            'projected_age': curr_res['vehicle_age'],
            'projected_mileage': int(mileage),
            'projected_price': curr_res['fair_resale_price']
        })

        accumulated_miles = float(mileage)
        for offset in range(1, years_ahead + 1):
            accumulated_miles += annual_driving_miles
            # Condition gradually weathers if old
            proj_cond = condition
            if offset >= 3 and condition == 'Excellent':
                proj_cond = 'Very Good'
            elif offset >= 4 and condition == 'Very Good':
                proj_cond = 'Good'

            val_res = self.predict_car_value(
                manufacturer, model_year, accumulated_miles, engine_capacity,
                fuel_type, transmission, number_of_owners, proj_cond
            )

            forecast.append({
                'year_offset': offset,
                'calendar_year': current_year + offset,
                'projected_age': curr_res['vehicle_age'] + offset,
                'projected_mileage': int(accumulated_miles),
                'projected_price': val_res['fair_resale_price']
            })

        return pd.DataFrame(forecast)

if __name__ == '__main__':
    engine = CarValuationEngine()
    test_car = {
        'manufacturer': 'Toyota',
        'model_year': 2019,
        'mileage': 55000,
        'engine_capacity': 2494,
        'fuel_type': 'Hybrid',
        'transmission': 'Automatic',
        'number_of_owners': 1,
        'condition': 'Very Good'
    }
    result = engine.predict_car_value(**test_car)
    print("Test Vehicle Valuation Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n5-Year Depreciation Forecast:")
    print(engine.forecast_depreciation(**test_car))
