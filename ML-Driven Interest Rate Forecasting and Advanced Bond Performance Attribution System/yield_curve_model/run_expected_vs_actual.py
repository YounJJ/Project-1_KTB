import pandas as pd
import numpy as np
from data_loader import DataLoader
from models import NSSModel, ForecastingModel, BondPricing

def run_analysis():
    print("--- Expected vs Actual P/L Analysis ---")
    
    loader = DataLoader()
    
    # 1. Load Data
    print("Loading Yields...")
    yields_df = loader.load_market_data(start_date='2017-01-01', end_date='2025-12-31')
    
    # Define Period (H2 2025)
    t0_date = pd.Timestamp('2025-06-30')
    t1_date = pd.Timestamp('2025-12-30')
    
    # Find nearest available dates
    t0_idx = yields_df.index.get_indexer([t0_date], method='nearest')[0]
    t1_idx = yields_df.index.get_indexer([t1_date], method='nearest')[0]
    t0_date = yields_df.index[t0_idx] 
    t1_date = yields_df.index[t1_idx]
    
    print(f"Analysis Period: {t0_date.date()} -> {t1_date.date()}")
    
    # Load Real Prices
    print("Loading Real Bond Prices from Oracle...")
    prices_10y = loader.load_bond_prices('10y', start_date=t0_date, end_date=t1_date)
    prices_50y = loader.load_bond_prices('50y', start_date=t0_date, end_date=t1_date)
    
    p_10y_actual_t0 = prices_10y.loc[t0_date, 'fair_value']
    p_10y_actual_t1 = prices_10y.loc[t1_date, 'fair_value']
    
    p_50y_actual_t0 = prices_50y.loc[t0_date, 'fair_value']
    p_50y_actual_t1 = prices_50y.loc[t1_date, 'fair_value']
    
    # 2. Train Model (Up to T0)
    print("Training Forecasting Model (Tenor-Specific XGBoost)...")
    train_df = yields_df.loc[:t0_date]
    model = ForecastingModel(lags=2)
    model.train(train_df)
    
    # 3. Forecast T1 (One-Step Ahead Strategy)
    # User Request: "Predict yield for target_date (T) using features from T-1" (Delta approach)
    # This aligns with the validation accuracy.
    
    print(f"Forecasting T1 ({t1_date.date()}) using T-1 Data...")
    
    # Locate T1 index
    # We want to use history ending at T1-1 to predict T1
    # Check if T1 is in df
    if t1_date not in yields_df.index:
         raise ValueError(f"Target date {t1_date} not found in yields_df")
         
    t1_loc = yields_df.index.get_loc(t1_date)
    # History up to T1-1 (iloc excludes endpoint in slicing if we use :t1_loc actually?)
    # yields_df.iloc[:t1_loc] gives rows 0 to t1_loc-1. Correct.
    # So the last row is T1-1.
    
    context_data = yields_df.iloc[:t1_loc]
    
    # Predict T1
    # models.predict_next_step handles:
    # 1. Calculate diffs from context
    # 2. Predict Delta
    # 3. Add Delta to Last_Level (T1-1)
    # Result is Expected Yields at T1
    final_forecast = model.predict_next_step(context_data)
    
    # 4. Construct Curves (NSS)
    tenors = [1, 3, 5, 10, 20, 30, 50]
    
    # Curve T0 (Actual)
    yields_t0 = yields_df.loc[t0_date, ['tenor_1y', 'tenor_3y', 'tenor_5y', 'tenor_10y', 'tenor_20y', 'tenor_30y', 'tenor_50y']].values
    nss_t0 = NSSModel()
    nss_t0.calibrate(tenors, yields_t0)
    
    # Curve T1 (Expected)
    # Map forecast series to array
    yields_t1_forecast = [final_forecast[f'tenor_{t}y'] for t in tenors]
    nss_t1_expected = NSSModel()
    nss_t1_expected.calibrate(tenors, yields_t1_forecast)
    
    # 5. Define Real Bonds
    # 10Y: Issue 2025-06-10, Mat 2035-06-10, Cpn 2.625%
    bond_10y = BondPricing(coupon_rate=0.02625, maturity_years=10.0, face_value=10000)
    # Adjust maturity based on T0 date relative to Issue? 
    # T0 is 2025-06-30. Issue 2025-06-10.
    # Maturity is fixed date: 2035-06-10.
    # Time to maturity at T0 = (2035-06-10 - 2025-06-30) in years.
    mat_date_10y = pd.Timestamp('2035-06-10')
    mat_years_10y_t0 = (mat_date_10y - t0_date).days / 365.0
    bond_10y.maturity_years = mat_years_10y_t0
    
    # 50Y: Issue 2024-09-10, Mat 2074-09-10, Cpn 2.750%
    bond_50y = BondPricing(coupon_rate=0.0275, maturity_years=50.0, face_value=10000)
    mat_date_50y = pd.Timestamp('2074-09-10')
    mat_years_50y_t0 = (mat_date_50y - t0_date).days / 365.0
    bond_50y.maturity_years = mat_years_50y_t0
    
    
    # 6. Calculate P/L
    dt_years = (t1_date - t0_date).days / 365.0
    
    # --- Expected P/L ---
    # Price at T0 (Theoretical - check vs Actual?)
    p_10y_theo_t0 = bond_10y.calculate_price(nss_t0)
    p_50y_theo_t0 = bond_50y.calculate_price(nss_t0)
    
    # Price at T1 (Expected)
    # Decrease maturity by dt
    bond_10y.maturity_years -= dt_years
    bond_50y.maturity_years -= dt_years
    
    p_10y_theo_t1 = bond_10y.calculate_price(nss_t1_expected)
    p_50y_theo_t1 = bond_50y.calculate_price(nss_t1_expected)
    
    # Coupon Income
    inc_10y = bond_10y.coupon_rate * 10000 * dt_years
    inc_50y = bond_50y.coupon_rate * 10000 * dt_years
    
    expected_pl_10y = (p_10y_theo_t1 - p_10y_theo_t0) + inc_10y
    expected_pl_50y = (p_50y_theo_t1 - p_50y_theo_t0) + inc_50y
    
    # --- Actual P/L ---
    # Using Oracle Prices directly
    actual_pl_10y = (p_10y_actual_t1 - p_10y_actual_t0) + inc_10y
    actual_pl_50y = (p_50y_actual_t1 - p_50y_actual_t0) + inc_50y
    
    # 7. Report
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'10Y Bond':<15} | {'50Y Bond':<15}")
    print("-" * 60)
    print(f"{'Expected P/L':<20} | {expected_pl_10y:,.2f}          | {expected_pl_50y:,.2f}")
    print(f"{'Actual P/L':<20} | {actual_pl_10y:,.2f}          | {actual_pl_50y:,.2f}")
    print(f"{'Difference':<20} | {(expected_pl_10y - actual_pl_10y):,.2f}          | {(expected_pl_50y - actual_pl_50y):,.2f}")
    
    acc_10y = "CORRECT" if np.sign(expected_pl_10y) == np.sign(actual_pl_10y) else "WRONG"
    acc_50y = "CORRECT" if np.sign(expected_pl_50y) == np.sign(actual_pl_50y) else "WRONG"
    
    print(f"{'Direction Acc':<20} | {acc_10y:<15} | {acc_50y:<15}")
    print("="*60)
    
    # 8. Visualization: Forecasted vs Realized Yield Curve
    print("Generating Yield Curve Comparison Plot...")
    
    import matplotlib.pyplot as plt
    
    # Get Actual Yields at T1
    # We already have yields_df loaded
    yields_t1_actual = yields_df.loc[t1_date, [f'tenor_{t}y' for t in tenors]].values
    
    # Calibrate NSS for Actual Curve
    nss_t1_actual = NSSModel()
    nss_t1_actual.calibrate(tenors, yields_t1_actual)
    
    # Generate Smooth Curves
    t_smooth = np.linspace(0.1, 50, 200)
    y_forecast_smooth = nss_t1_expected.get_spot_rate(t_smooth)
    y_actual_smooth = nss_t1_actual.get_spot_rate(t_smooth)
    
    plt.figure(figsize=(12, 8))
    
    # Plot Curves
    plt.plot(t_smooth, y_forecast_smooth, label='Forecasted Curve (Dec 2025)', linestyle='--', color='blue', linewidth=2)
    plt.plot(t_smooth, y_actual_smooth, label='Realized Curve (Dec 2025)', linestyle='-', color='red', linewidth=2)
    
    # Plot Points
    plt.scatter(tenors, yields_t1_forecast, color='blue', marker='x', s=100, label='Forecasted Points', zorder=5)
    plt.scatter(tenors, yields_t1_actual, color='red', marker='o', s=100, label='Realized Points', zorder=5)
    
    plt.title(f"Yield Curve Comparison: Expected vs Actual ({t1_date.date()})", fontsize=16)
    plt.xlabel("Maturity (Years)", fontsize=12)
    plt.ylabel("Spot Rate (%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expected_vs_actual_curve.png')
    print("Plot saved to 'expected_vs_actual_curve.png'")
    
    # 9. Save Table as PNG
    print("Saving P/L Table as PNG...")
    
    # Prepare Data
    data = [
        ["Expected P/L", f"{expected_pl_10y:,.2f}", f"{expected_pl_50y:,.2f}"],
        ["Actual P/L", f"{actual_pl_10y:,.2f}", f"{actual_pl_50y:,.2f}"],
        ["Difference", f"{(expected_pl_10y - actual_pl_10y):,.2f}", f"{(expected_pl_50y - actual_pl_50y):,.2f}"],
        ["Direction Acc", acc_10y, acc_50y]
    ]
    columns = ["Metric", "10Y Bond", "50Y Bond"]
    
    # Create Figure for Table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Create Table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    
    # Style Table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Scale width and height
    
    # Header Formatting
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_edgecolor('#dddddd')
    
    plt.title(f"Expected vs Actual P/L Analysis ({t1_date.date()})", fontsize=14, weight='bold', pad=20)
    plt.savefig('expected_vs_actual_table.png', bbox_inches='tight', dpi=300)
    print("Table saved to 'expected_vs_actual_table.png'")

if __name__ == "__main__":
    run_analysis()
