"""
Feature Engineering & Data Pipeline Module
Implements domain-specific automotive features and scikit-learn transformers.
Strict featurization ordering: split data before fitting transformers!
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

CURRENT_YEAR = 2024

BRAND_TIERS = {
    'Nissan': 'Economy',
    'Hyundai': 'Economy',
    'Toyota': 'Mainstream',
    'Honda': 'Mainstream',
    'Ford': 'Mainstream',
    'Chevrolet': 'Mainstream',
    'Volkswagen': 'Mainstream',
    'Audi': 'Luxury',
    'BMW': 'Luxury',
    'Mercedes-Benz': 'Luxury'
}

CONDITION_ORDER = ['Fair', 'Good', 'Very Good', 'Excellent']

class AutomotiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs high-value domain features for used vehicle valuation:
    - vehicle_age: Years since manufacture
    - annual_mileage: Odometer miles normalized by age
    - mileage_vs_expected: Deviation from national baseline of 12,000 miles/year
    - engine_tier: Displacement categorization
    - brand_tier: Economic, Mainstream, or Luxury brand prestige
    - wear_index: Holistic wear and tear composite score
    """
    def __init__(self, current_year=CURRENT_YEAR):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Vehicle Age
        df['vehicle_age'] = np.maximum(0, self.current_year - df['model_year'])
        
        # 2. Annual Mileage (miles driven per year)
        effective_age = np.maximum(df['vehicle_age'], 0.5)
        df['annual_mileage'] = df['mileage'] / effective_age
        
        # 3. Deviation from typical expected mileage (12,000 miles/yr)
        expected_miles = effective_age * 12000.0
        df['mileage_vs_expected'] = df['mileage'] - expected_miles
        
        # 4. Brand Tier Mapping
        df['brand_tier'] = df['manufacturer'].map(BRAND_TIERS).fillna('Mainstream')
        
        # 5. Engine Capacity Tier
        def get_engine_tier(cc):
            if cc == 0:
                return 'Zero_Electric'
            elif cc < 1400:
                return 'SubCompact'
            elif cc <= 2200:
                return 'Compact_Midsize'
            elif cc <= 3500:
                return 'Large_V6'
            else:
                return 'Performance_V8'
                
        df['engine_tier'] = df['engine_capacity'].apply(get_engine_tier)
        
        # 6. Holistic Wear & Tear Composite Index
        # Combines age, owners, and annual mileage stress
        df['wear_index'] = (
            (df['vehicle_age'] * 0.40) + 
            (df['number_of_owners'] * 0.70) + 
            (df['annual_mileage'] / 10000.0 * 0.35)
        )
        
        return df

def build_preprocessor():
    """
    Builds the scikit-learn ColumnTransformer for preprocessing.
    - Categorical features: One-Hot Encoded
    - Condition: Ordinal Encoded (Fair < Good < Very Good < Excellent)
    - Numerical features: Standard Scaled
    """
    categorical_features = ['manufacturer', 'fuel_type', 'transmission', 'brand_tier', 'engine_tier']
    ordinal_features = ['condition']
    numerical_features = [
        'model_year', 'mileage', 'engine_capacity', 'number_of_owners',
        'vehicle_age', 'annual_mileage', 'mileage_vs_expected', 'wear_index'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('ord', OrdinalEncoder(categories=[CONDITION_ORDER]), ordinal_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def create_full_pipeline(regressor):
    """
    Chains domain feature engineering, preprocessing, and the regressor model.
    """
    return Pipeline([
        ('feature_engineer', AutomotiveFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', regressor)
    ])

