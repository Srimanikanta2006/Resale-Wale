"""
Dataset Generator & Real-World Automotive Benchmark Synthesizer
Generates a realistic, statistically grounded used car marketplace dataset
strictly adhering to the required features:
- manufacturer
- model_year
- mileage
- engine_capacity
- fuel_type
- transmission
- number_of_owners
- condition
- resale_price (Target)
"""

import numpy as np
import pandas as pd
import os

def generate_used_car_dataset(n_samples=25000, random_seed=42):
    np.random.seed(random_seed)

    # 1. Manufacturers & Market Tiers
    manufacturers_config = {
        # Brand: (Market Tier, Base MSRP Mean, Base MSRP Std, Annual Depreciation Rate)
        'Toyota': ('Mainstream_Reliable', 28000, 6000, 0.085),
        'Honda': ('Mainstream_Reliable', 27000, 5500, 0.088),
        'Ford': ('Mainstream_Domestic', 31000, 8000, 0.105),
        'Chevrolet': ('Mainstream_Domestic', 29000, 7500, 0.110),
        'Nissan': ('Economy', 24000, 5000, 0.118),
        'Hyundai': ('Economy_Value', 23500, 5000, 0.102),
        'Volkswagen': ('Mainstream_European', 29000, 6500, 0.112),
        'BMW': ('Luxury_European', 56000, 14000, 0.138),
        'Mercedes-Benz': ('Luxury_European', 62000, 16000, 0.135),
        'Audi': ('Luxury_European', 54000, 13000, 0.132),
    }
    
    brand_list = list(manufacturers_config.keys())
    brand_probs = [0.16, 0.14, 0.13, 0.12, 0.09, 0.11, 0.08, 0.06, 0.06, 0.05]
    
    selected_brands = np.random.choice(brand_list, size=n_samples, p=brand_probs)

    # 2. Model Year (2008 to 2024, current year 2024 reference)
    CURRENT_YEAR = 2024
    # Distribution biased towards 3-10 year old cars (typical marketplace sweet spot)
    year_weights = np.array([
        0.015, 0.02, 0.025, 0.03, 0.035, 0.045, # 2008-2013
        0.055, 0.065, 0.075, 0.085, 0.095, 0.11, # 2014-2019
        0.10, 0.09, 0.08, 0.055, 0.025           # 2020-2024
    ])
    year_weights /= year_weights.sum()
    years = np.arange(2008, 2025)
    model_years = np.random.choice(years, size=n_samples, p=year_weights)
    vehicle_ages = CURRENT_YEAR - model_years

    # 3. Fuel Types
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
    fuel_probs = [0.65, 0.18, 0.12, 0.05]
    selected_fuels = np.random.choice(fuel_types, size=n_samples, p=fuel_probs)

    # 4. Engine Capacity (cc)
    # Electric cars don't have displacement (represented as 0 cc or equivalent battery displacement)
    engine_displacements = []
    for brand, fuel in zip(selected_brands, selected_fuels):
        if fuel == 'Electric':
            engine_displacements.append(0)
        else:
            tier = manufacturers_config[brand][0]
            if 'Luxury' in tier:
                # Luxury: 2.0L to 4.4L (1998cc to 4395cc)
                cc_choices = [1998, 2498, 2993, 2998, 3982, 4395]
                p_cc = [0.30, 0.20, 0.25, 0.15, 0.06, 0.04]
            elif 'Domestic' in tier:
                # Ford/Chevy: trucks/SUVs/V6/V8 or small turbo
                cc_choices = [1498, 1999, 2488, 3496, 5038]
                p_cc = [0.20, 0.35, 0.20, 0.15, 0.10]
            else:
                # Asian/European mainstream: 1.0L to 2.5L
                cc_choices = [999, 1197, 1498, 1598, 1798, 1998, 2494]
                p_cc = [0.10, 0.15, 0.25, 0.15, 0.15, 0.12, 0.08]
            engine_displacements.append(int(np.random.choice(cc_choices, p=p_cc)))
    
    engine_capacities = np.array(engine_displacements)

    # 5. Transmission
    # Luxury mostly Automatic; Mainstream mix with majority Automatic in modern era
    transmissions = []
    for brand, year in zip(selected_brands, model_years):
        auto_prob = 0.95 if 'Luxury' in manufacturers_config[brand][0] else (0.75 + (year - 2008) * 0.012)
        auto_prob = min(0.97, max(0.65, auto_prob))
        trans = 'Automatic' if np.random.rand() < auto_prob else 'Manual'
        transmissions.append(trans)
    transmissions = np.array(transmissions)

    # 6. Mileage (Odometer in miles)
    # Average driver drives ~12,000 miles per year, with lognormal variance
    base_annual_mileage = np.random.normal(12200, 3200, size=n_samples)
    base_annual_mileage = np.clip(base_annual_mileage, 3500, 26000)
    # Young cars have non-zero delivery/test miles
    effective_age = np.maximum(vehicle_ages, 0.4)
    mileages = np.round(effective_age * base_annual_mileage + np.random.exponential(1500, size=n_samples))
    # Cap mileage realistically
    mileages = np.clip(mileages, 500, 280000).astype(int)

    # 7. Number of Owners (1 to 4+)
    # Older cars naturally accumulate more owners
    num_owners_list = []
    for age in vehicle_ages:
        if age <= 2:
            p_own = [0.92, 0.07, 0.01, 0.0]
        elif age <= 5:
            p_own = [0.65, 0.28, 0.06, 0.01]
        elif age <= 9:
            p_own = [0.35, 0.42, 0.17, 0.06]
        else:
            p_own = [0.15, 0.35, 0.32, 0.18]
        own = np.random.choice([1, 2, 3, 4], p=p_own)
        num_owners_list.append(own)
    number_of_owners = np.array(num_owners_list)

    # 8. Condition (Fair, Good, Very Good, Excellent)
    # Correlated with age, owners, and mileage deviation
    conditions = []
    condition_choices = ['Fair', 'Good', 'Very Good', 'Excellent']
    for age, own, miles in zip(vehicle_ages, number_of_owners, mileages):
        miles_per_year = miles / max(age, 1)
        wear_score = (age * 0.35) + (own * 0.7) + (miles_per_year / 10000.0 * 0.4) + np.random.normal(0, 0.6)
        if wear_score < 2.0:
            cond = 'Excellent'
        elif wear_score < 3.8:
            cond = 'Very Good'
        elif wear_score < 5.8:
            cond = 'Good'
        else:
            cond = 'Fair'
        conditions.append(cond)
    conditions = np.array(conditions)

    # 9. Realistic Resale Price Calculation (Ground Truth Economic Engine)
    resale_prices = []
    for i in range(n_samples):
        brand = selected_brands[i]
        year = model_years[i]
        age = vehicle_ages[i]
        miles = mileages[i]
        cc = engine_capacities[i]
        fuel = selected_fuels[i]
        trans = transmissions[i]
        owners = number_of_owners[i]
        cond = conditions[i]

        _, base_msrp_mean, base_msrp_std, depr_rate = manufacturers_config[brand]
        
        # Original MSRP estimate
        msrp = np.random.normal(base_msrp_mean, base_msrp_std * 0.35)
        # Engine capacity premium
        if cc > 0:
            msrp *= (1.0 + (cc - 1800) / 4500.0 * 0.40)
        else: # Electric powertrain premium
            msrp *= 1.22
        
        if fuel == 'Hybrid':
            msrp *= 1.08
        elif fuel == 'Diesel':
            msrp *= 1.04

        if trans == 'Automatic':
            msrp += 1200

        # Depreciation factors
        # A) Age depreciation: exponential decay curve
        # Luxury drops fast initially, Toyota/Honda retain value
        age_factor = np.exp(-depr_rate * age)
        
        # B) Mileage adjustment (Benchmark is 12,000 miles/year)
        expected_miles = max(age, 0.5) * 12000
        mileage_diff = miles - expected_miles
        # Penalize excess miles, slight reward for low miles
        if mileage_diff > 0:
            mileage_multiplier = max(0.55, 1.0 - (mileage_diff / 150000.0) * 0.38)
        else:
            mileage_multiplier = min(1.20, 1.0 + (abs(mileage_diff) / 100000.0) * 0.15)

        # C) Owner penalty
        owner_multipliers = {1: 1.0, 2: 0.94, 3: 0.87, 4: 0.80}
        owner_factor = owner_multipliers[owners]

        # D) Condition multiplier
        condition_multipliers = {'Excellent': 1.12, 'Very Good': 1.03, 'Good': 0.91, 'Fair': 0.76}
        condition_factor = condition_multipliers[cond]

        # Combine
        price = msrp * age_factor * mileage_multiplier * owner_factor * condition_factor

        # Real-world market noise (+- 5.5% standard deviation)
        market_noise = np.random.normal(1.0, 0.055)
        price *= market_noise

        # Operational salvage / scrap floor value
        floor_value = 1800 if cond == 'Fair' else 2800
        price = max(floor_value, price)

        resale_prices.append(round(price, 2))

    df = pd.DataFrame({
        'manufacturer': selected_brands,
        'model_year': model_years,
        'mileage': mileages,
        'engine_capacity': engine_capacities,
        'fuel_type': selected_fuels,
        'transmission': transmissions,
        'number_of_owners': number_of_owners,
        'condition': conditions,
        'resale_price': resale_prices
    })

    return df

if __name__ == '__main__':
    print("Generating authentic automotive marketplace dataset...")
    df = generate_used_car_dataset(n_samples=25000, random_seed=42)
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'used_cars_marketplace.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(df.head(10))
    print("\nSummary Statistics:")
    print(df.describe())

