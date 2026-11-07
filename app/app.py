"""
Resale Wale - AI Automobile Resale Price Valuation Platform
An enterprise-grade, interactive marketplace ML valuation system.
Estimates resale prices based on:
- Manufacturer
- Model Year
- Mileage
- Engine Capacity
- Fuel Type
- Transmission
- Number of Owners
- Condition
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src modules are discoverable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import CarValuationEngine
from feature_engineering import BRAND_TIERS, CONDITION_ORDER

# Page configuration
st.set_page_config(
    page_title="Resale Wale | AI Car Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .price-hero {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0F766E;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-economy { background-color: #DCFCE7; color: #166534; }
    .badge-mainstream { background-color: #DBEAFE; color: #1E40AF; }
    .badge-luxury { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_valuation_engine():
    model_path = os.path.join(root_dir, 'models', 'car_valuation_pipeline.joblib')
    return CarValuationEngine(model_path=model_path)

@st.cache_data
def load_metadata_and_reports():
    reports_dir = os.path.join(root_dir, 'reports')
    
    benchmark_path = os.path.join(reports_dir, 'model_benchmark_results.csv')
    benchmark_df = pd.read_csv(benchmark_path) if os.path.exists(benchmark_path) else None
    
    summary_path = os.path.join(reports_dir, 'test_evaluation_summary.json')
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
        
    return benchmark_df, summary_data

engine = load_valuation_engine()
benchmark_df, summary_data = load_metadata_and_reports()

# Header
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🚗 Resail Wale - Used Car Valuation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production ML Pricing Intelligence powered by LightGBM & Domain Feature Engineering</div>', unsafe_allow_html=True)
with col_stat:
    test_mape = summary_data['metrics']['MAPE (%)']
    test_r2 = summary_data['metrics']['R2']
    st.metric(label="Model Accuracy (Holdout Test)", value=f"{test_r2:.4f} R²", delta=f"{test_mape}% MAPE")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Vehicle Valuation Studio",
    "📈 5-Year Depreciation & Dynamics",
    "📊 Marketplace Insights (EDA)",
    "🧠 Model Governance & Benchmarks"
])

# -------------------------------------------------------------
# TAB 1: VEHICLE VALUATION STUDIO
# -------------------------------------------------------------
with tab1:
    st.subheader("Enter Vehicle Specifications")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        manufacturer = st.selectbox(
            "Manufacturer (Brand)",
            options=['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Hyundai', 'Volkswagen', 'BMW', 'Mercedes-Benz', 'Audi'],
            index=0
        )
        model_year = st.slider("Model Year", min_value=2008, max_value=2024, value=2019, step=1)
        mileage = st.number_input("Odometer Mileage (miles)", min_value=500, max_value=300000, value=52000, step=1000)

    with col_in2:
        fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'Hybrid', 'Electric'], index=0)
        
        default_cc = 0 if fuel_type == 'Electric' else 1998
        engine_capacity = st.number_input(
            "Engine Capacity (cc) [0 for EV]",
            min_value=0, max_value=6000,
            value=default_cc, step=100,
            help="Engine displacement in cubic centimeters. Enter 0 for Electric vehicles."
        )
        transmission = st.selectbox("Transmission", options=['Automatic', 'Manual'], index=0)

    with col_in3:
        number_of_owners = st.selectbox(
            "Number of Previous Owners",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"{x} Owner" if x == 1 else f"{x} Owners",
            index=0
        )
        condition = st.selectbox(
            "Physical & Mechanical Condition",
            options=['Fair', 'Good', 'Very Good', 'Excellent'],
            index=2,
            help="Fair: Minor cosmetic/mechanical needs; Good: Normal wear; Very Good: Well maintained; Excellent: Pristine showroom quality"
        )
        
        estimate_btn = st.button("🔍 Calculate Fair Resale Value", type="primary", use_container_width=True)

    # Perform Valuation
    val_res = engine.predict_car_value(
        manufacturer=manufacturer,
        model_year=model_year,
        mileage=mileage,
        engine_capacity=engine_capacity,
        fuel_type=fuel_type,
        transmission=transmission,
        number_of_owners=number_of_owners,
        condition=condition
    )

    st.markdown("---")
    st.subheader("Automobile Market Valuation Report")

    col_res1, col_res2, col_res3 = st.columns([1.5, 1, 1])

    with col_res1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #0F766E;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #475569; text-transform: uppercase;">Fair Purchase Price</div>
            <div class="price-hero">${val_res['fair_resale_price']:,.2f}</div>
            <div style="font-size: 0.88rem; color: #64748B; margin-top: 6px;">
                <b>95% Prediction Interval:</b> ${val_res['prediction_interval_95'][0]:,.2f} &ndash; ${val_res['prediction_interval_95'][1]:,.2f}
            </div>
            <div style="margin-top: 10px;">
                <span class="badge-pill badge-{val_res['brand_tier'].lower()}">{val_res['brand_tier']} Brand Tier</span>
                <span class="badge-pill" style="background:#F1F5F9; color:#334155;">Age: {val_res['vehicle_age']} Years</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #2563EB;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">🏷️ DEALER TRADE-IN</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #1E40AF; margin: 8px 0;">${val_res['trade_in_value']:,.2f}</div>
            <div style="font-size: 0.8rem; color: #64748B;">Instant dealer cash offer / wholesale baseline (~88% fair market)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_res3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #7C3AED;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">🏬 DEALER RETAIL VALUE</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #6D28D9; margin: 8px 0;">${val_res['dealer_retail_value']:,.2f}</div>
            <div style="font-size: 0.8rem; color: #64748B;">Dealer inventory sticker price with inspection & warranty (~112%)</div>
        </div>
        """, unsafe_allow_html=True)

    # Key Vehicle Health Diagnostic
    st.markdown("#### Vehicle Usage & Condition Assessment")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Annual Mileage Rate", f"{mileage / max(val_res['vehicle_age'], 0.5):,.0f} mi/yr", 
                  f"{val_res['mileage_delta']:+,.0f} vs expected")
    with d2:
        st.metric("Odometer Status", "Normal Commuter" if abs(val_res['mileage_delta']) < 15000 else ("Low Mileage Premium" if val_res['mileage_delta'] < 0 else "High Mileage Wear"))
    with d3:
        st.metric("Ownership Multiplier", "1.00x (Baseline)" if number_of_owners == 1 else f"~{0.94 if number_of_owners==2 else (0.87 if number_of_owners==3 else 0.80):.2f}x")
    with d4:
        st.metric("Condition Grade", condition, f"Grade {CONDITION_ORDER.index(condition)+1} of 4")

# -------------------------------------------------------------
# TAB 2: 5-YEAR DEPRECIATION & DYNAMICS
# -------------------------------------------------------------
with tab2:
    st.subheader("5-Year Future Depreciation Projections")
    st.markdown("Visualizes how this vehicle's fair market value will evolve over the next 5 years based on average annual mileage accumulation (12,000 miles/year) and aging curves.")
    
    annual_miles_proj = st.slider("Projected Annual Driving (miles/year)", min_value=5000, max_value=25000, value=12000, step=1000)
    
    forecast_df = engine.forecast_depreciation(
        manufacturer=manufacturer,
        model_year=model_year,
        mileage=mileage,
        engine_capacity=engine_capacity,
        fuel_type=fuel_type,
        transmission=transmission,
        number_of_owners=number_of_owners,
        condition=condition,
        years_ahead=5,
        annual_driving_miles=annual_miles_proj
    )

    c_chart, c_table = st.columns([2, 1])

    with c_chart:
        fig_dep, ax_dep = plt.subplots(figsize=(9, 4.8))
        ax_dep.plot(forecast_df['calendar_year'], forecast_df['projected_price'], marker='o', lw=2.5, color='#0F766E')
        for _, row in forecast_df.iterrows():
            ax_dep.annotate(
                f"${row['projected_price']:,.0f}",
                (row['calendar_year'], row['projected_price']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontweight='bold',
                fontsize=9
            )
        ax_dep.set_title(f'Estimated Fair Market Value: {model_year} {manufacturer}', fontsize=12, fontweight='bold')
        ax_dep.set_xlabel('Calendar Year', fontsize=10)
        ax_dep.set_ylabel('Projected Value ($)', fontsize=10)
        ax_dep.yaxis.set_major_formatter('${x:,.0f}')
        ax_dep.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_dep)
        plt.close(fig_dep)

    with c_table:
        st.markdown("**Depreciation Schedule Table**")
        display_tbl = forecast_df[['calendar_year', 'projected_age', 'projected_mileage', 'projected_price']].copy()
        display_tbl.columns = ['Year', 'Age', 'Odometer (mi)', 'Estimated Value ($)']
        display_tbl['Estimated Value ($)'] = display_tbl['Estimated Value ($)'].apply(lambda x: f"${x:,.2f}")
        display_tbl['Odometer (mi)'] = display_tbl['Odometer (mi)'].apply(lambda x: f"{x:,}")
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: MARKETPLACE INSIGHTS (EDA)
# -------------------------------------------------------------
with tab3:
    st.subheader("Marketplace Data Profiling & Domain Patterns")
    st.markdown("Exploratory Data Analysis across 25,000 real-world automotive marketplace transactions.")

    eda_fig_col1, eda_fig_col2 = st.columns(2)
    
    with eda_fig_col1:
        f1_path = os.path.join(root_dir, 'reports', 'figures', '02_depreciation_by_brand_tier.png')
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Depreciation Rates across Economy, Mainstream, and Luxury Brands", use_container_width=True)
            
        f3_path = os.path.join(root_dir, 'reports', 'figures', '04_condition_and_owners_impact.png')
        if os.path.exists(f3_path):
            st.image(f3_path, caption="Ownership Count Penalties & Condition Distribution", use_container_width=True)

    with eda_fig_col2:
        f2_path = os.path.join(root_dir, 'reports', 'figures', '03_mileage_vs_condition.png')
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Resale Price vs. Mileage Stratified by Physical Condition", use_container_width=True)

        f4_path = os.path.join(root_dir, 'reports', 'figures', '06_correlation_heatmap.png')
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Feature Correlation Heatmap (Raw & Engineered Domain Features)", use_container_width=True)

# -------------------------------------------------------------
# TAB 4: MODEL GOVERNANCE & BENCHMARKS
# -------------------------------------------------------------
with tab4:
    st.subheader("Model Benchmarking & Validation Rigor")
    st.markdown("""
    In accordance with strict Machine Learning best practices:
    - **Featurization Ordering**: Training, validation, and test datasets were split prior to fitting transformers.
    - **Target Transformation**: The continuous target `resale_price` was modeled with log1p transformation to ensure variance stabilization and positive predictions.
    - **Cross-Validation**: 5-fold cross validation was conducted across training data to verify stability.
    - **Bootstrap Confidence Intervals**: 1,000 iterations were run to compute 95% confidence intervals on holdout test error.
    """)

    st.markdown("#### Model Benchmark Comparison Table")
    if benchmark_df is not None:
        st.dataframe(
            benchmark_df.style.highlight_min(subset=['MAPE (%)', 'MAE', 'RMSE'], color='#DCFCE7')
                       .highlight_max(subset=['R2'], color='#DCFCE7'),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("#### Holdout Test Set Performance & 95% Confidence Intervals")
    ci = summary_data['confidence_intervals_95']
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test R² Score", f"{summary_data['metrics']['R2']:.4f}")
    with m_col2:
        st.metric("Test MAPE", f"{summary_data['metrics']['MAPE (%)']:.2f}%", f"95% CI: [{ci['MAPE']['lower']}%, {ci['MAPE']['upper']}%]")
    with m_col3:
        st.metric("Test MAE", f"${summary_data['metrics']['MAE']:,.2f}", f"95% CI: [${ci['MAE']['lower']:,.0f}, ${ci['MAE']['upper']:,.0f}]")
    with m_col4:
        st.metric("Test RMSE", f"${summary_data['metrics']['RMSE']:,.2f}", f"95% CI: [${ci['RMSE']['lower']:,.0f}, ${ci['RMSE']['upper']:,.0f}]")

    st.markdown("#### Slice-Based Error Analysis")
    sl_col1, sl_col2, sl_col3 = st.columns(3)
    with sl_col1:
        st.markdown("**By Vehicle Age Bracket**")
        p_age = os.path.join(root_dir, 'reports', 'slice_analysis_age_bracket.csv')
        if os.path.exists(p_age):
            st.dataframe(pd.read_csv(p_age), use_container_width=True, hide_index=True)
            
    with sl_col2:
        st.markdown("**By Brand Market Tier**")
        p_tier = os.path.join(root_dir, 'reports', 'slice_analysis_brand_tier.csv')
        if os.path.exists(p_tier):
            st.dataframe(pd.read_csv(p_tier), use_container_width=True, hide_index=True)
            
    with sl_col3:
        st.markdown("**By Vehicle Condition**")
        p_cond = os.path.join(root_dir, 'reports', 'slice_analysis_condition.csv')
        if os.path.exists(p_cond):
            st.dataframe(pd.read_csv(p_cond), use_container_width=True, hide_index=True)

    st.markdown("#### Interpretability & Residual Diagnostics")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        f_res = os.path.join(root_dir, 'reports', 'figures', '07_test_actual_vs_predicted_residuals.png')
        if os.path.exists(f_res):
            st.image(f_res, caption="Test Set: Actual vs. Predicted & Residual Distribution", use_container_width=True)
            
    with res_col2:
        f_shap = os.path.join(root_dir, 'reports', 'figures', '08_shap_summary_plot.png')
        if os.path.exists(f_shap):
            st.image(f_shap, caption="Global SHAP Attribution (TreeExplainer)", use_container_width=True)

