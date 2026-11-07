"""
Exploratory Data Analysis & Visualizations for Automobile Resale Marketplace
Generates high-resolution publication-quality plots and statistical profiles:
1. Target Distribution & Log-Transformation (Normality & Skewness)
2. Depreciation Curves across Brand Tiers (Toyota/Honda vs BMW/Mercedes)
3. Mileage Impact & Annualized Wear Patterns
4. Physical Condition & Number of Owners Valuation Matrix
5. Correlation Heatmap of Raw & Engineered Features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import AutomotiveFeatureEngineer

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = os.path.join('reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def run_eda(data_path='data/used_cars_marketplace.csv'):
    print("Loading data for Exploratory Data Analysis...")
    df = pd.read_csv(data_path)
    
    # Apply domain feature engineering to analyze enriched features
    fe = AutomotiveFeatureEngineer()
    df_enriched = fe.transform(df)

    # -------------------------------------------------------------
    # 1. Target Distribution (Raw vs Log-Transformed)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['resale_price'], kde=True, ax=axes[0], color='#1f77b4', bins=40)
    axes[0].set_title('Raw Resale Price Distribution (Right-Skewed)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Resale Price ($)', fontsize=11)
    axes[0].set_ylabel('Vehicle Count', fontsize=11)
    median_val = df['resale_price'].median()
    axes[0].axvline(median_val, color='red', linestyle='--', label=f'Median: ${median_val:,.0f}')
    axes[0].legend()
    
    log_price = np.log1p(df['resale_price'])
    sns.histplot(log_price, kde=True, ax=axes[1], color='#2ca02c', bins=40)
    axes[1].set_title('Log-Transformed Resale Price (Normalized Variance)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('log(Resale Price + 1)', fontsize=11)
    axes[1].set_ylabel('Vehicle Count', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '01_target_price_distribution.png'), dpi=300)
    plt.close()
    print("-> Saved: 01_target_price_distribution.png")

    # -------------------------------------------------------------
    # 2. Depreciation Curves across Brand Tiers
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_enriched,
        x='model_year',
        y='resale_price',
        hue='brand_tier',
        marker='o',
        palette={'Economy': '#2ca02c', 'Mainstream': '#1f77b4', 'Luxury': '#d62728'},
        errorbar=('ci', 95),
        ax=ax
    )
    ax.set_title('Resale Price by Model Year across Brand Tiers (Depreciation Curves)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Year', fontsize=12)
    ax.set_ylabel('Resale Price ($)', fontsize=12)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '02_depreciation_by_brand_tier.png'), dpi=300)
    plt.close()
    print("-> Saved: 02_depreciation_by_brand_tier.png")

    # -------------------------------------------------------------
    # 3. Mileage & Condition Interaction
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df_enriched.sample(n=3500, random_state=42)
    sns.scatterplot(
        data=sample_df,
        x='mileage',
        y='resale_price',
        hue='condition',
        hue_order=['Fair', 'Good', 'Very Good', 'Excellent'],
        palette='viridis',
        alpha=0.65,
        s=30,
        ax=ax
    )
    ax.set_title('Resale Price vs. Mileage Stratified by Vehicle Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Odometer Mileage (miles)', fontsize=12)
    ax.set_ylabel('Resale Price ($)', fontsize=12)
    ax.xaxis.set_major_formatter('{x:,.0f}')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '03_mileage_vs_condition.png'), dpi=300)
    plt.close()
    print("-> Saved: 03_mileage_vs_condition.png")

    # -------------------------------------------------------------
    # 4. Boxplot of Condition & Number of Owners Impact
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(
        data=df_enriched,
        x='condition',
        y='resale_price',
        order=['Fair', 'Good', 'Very Good', 'Excellent'],
        palette='Blues',
        ax=axes[0]
    )
    axes[0].set_title('Resale Price across Condition Grades', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Resale Price ($)', fontsize=11)
    axes[0].yaxis.set_major_formatter('${x:,.0f}')

    sns.boxplot(
        data=df_enriched,
        x='number_of_owners',
        y='resale_price',
        palette='Reds_r',
        ax=axes[1]
    )
    axes[1].set_title('Ownership Penalty: Resale Price by Number of Owners', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Resale Price ($)', fontsize=11)
    axes[1].yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_condition_and_owners_impact.png'), dpi=300)
    plt.close()
    print("-> Saved: 04_condition_and_owners_impact.png")

    # -------------------------------------------------------------
    # 5. Engine Capacity & Fuel Type Interaction
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_enriched[df_enriched['fuel_type'] != 'Electric'],
        x='engine_tier',
        y='resale_price',
        hue='fuel_type',
        order=['SubCompact', 'Compact_Midsize', 'Large_V6', 'Performance_V8'],
        palette='Set2',
        ax=ax
    )
    ax.set_title('Valuation by Engine Displacement Tier & Fuel Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Engine Displacement Bracket', fontsize=12)
    ax.set_ylabel('Average Resale Price ($)', fontsize=12)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '05_engine_tier_fuel_type.png'), dpi=300)
    plt.close()
    print("-> Saved: 05_engine_tier_fuel_type.png")

    # -------------------------------------------------------------
    # 6. Correlation Heatmap
    # -------------------------------------------------------------
    numeric_cols = [
        'resale_price', 'model_year', 'mileage', 'engine_capacity',
        'number_of_owners', 'vehicle_age', 'annual_mileage',
        'mileage_vs_expected', 'wear_index'
    ]
    corr = df_enriched[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax, square=True, vmin=-1, vmax=1)
    ax.set_title('Pearson Correlation Heatmap (Raw & Engineered Numerical Features)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '06_correlation_heatmap.png'), dpi=300)
    plt.close()
    print("-> Saved: 06_correlation_heatmap.png")

    print("\nEDA completed successfully! Visualizations stored in reports/figures/")

if __name__ == '__main__':
    run_eda()

