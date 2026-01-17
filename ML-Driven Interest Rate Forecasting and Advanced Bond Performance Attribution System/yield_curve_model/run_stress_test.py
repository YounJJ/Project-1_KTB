import pandas as pd
import numpy as np
from data_loader import DataLoader
from models import NSSModel, BondPricing, ForecastingModel

def run_stress_test():
    loader = DataLoader()
    
    # Load Data (Full History until 2025-12-31 to get T1 actuals if needed, 
    # but for Stress Test we are simulating "What if we are at T0 and T1 happens?")
    # Actually, Stress Test usually assumes we are at T0 (Jun 2025) and applying a shock.
    # The user said: "Use the ML-forecasted yield curve for Dec 2025 as the stress scenario."
    
    print("Loading Data...")
    yields_df = loader.load_market_data(start_date='2017-01-01', end_date='2025-12-31')
    
    t0_date = pd.Timestamp('2025-06-30')
    train_df = yields_df.loc[:t0_date]
    
    print(f"Base Date (T0): {t0_date.date()}")
    
    # 1. Base Model (T0 Actual)
    tenors = [1, 3, 5, 10, 20, 30, 50]
    yields_t0 = yields_df.loc[t0_date, ['tenor_1y', 'tenor_3y', 'tenor_5y', 'tenor_10y', 'tenor_20y', 'tenor_30y', 'tenor_50y']].values
    
    nss_base = NSSModel()
    nss_base.calibrate(tenors, yields_t0)
    
    # 2. Define Assets (Real Bonds)
    # 10Y: Issue 2025-06-10, Mat 2035-06-10, Cpn 2.625%
    bond_10y = BondPricing(coupon_rate=0.02625, maturity_years=10.0, face_value=10000)
    mat_date_10y = pd.Timestamp('2035-06-10')
    bond_10y.maturity_years = (mat_date_10y - t0_date).days / 365.0
    
    # 50Y: Issue 2024-09-10, Mat 2074-09-10, Cpn 2.750%
    bond_50y = BondPricing(coupon_rate=0.0275, maturity_years=50.0, face_value=10000)
    mat_date_50y = pd.Timestamp('2074-09-10')
    bond_50y.maturity_years = (mat_date_50y - t0_date).days / 365.0
    
    # Portfolio (Assume 50/50 mix typically, or just sum)
    p_10y_base = bond_10y.calculate_price(nss_base)
    p_50y_base = bond_50y.calculate_price(nss_base)
    total_assets_base = p_10y_base + p_50y_base
    
    print(f"Total Assets (Base): {total_assets_base:,.2f}")
    
    # 3. Liability (ALM Setting)
    # Ratio 80%
    liability_value = total_assets_base * 0.80
    
    # Liability Duration?
    # K-ICS often assumes long liability duration. Let's keep 15y or estimate?
    # User didn't specify Liability Duration, just L/A ratio. Kept 15.0 from previous logic.
    liability_duration = 15.0 
    
    # Asset Duration
    dur_10y = bond_10y.calculate_effective_duration(nss_base)
    dur_50y = bond_50y.calculate_effective_duration(nss_base)
    
    w_10y = p_10y_base / total_assets_base
    w_50y = p_50y_base / total_assets_base
    asset_duration = w_10y * dur_10y + w_50y * dur_50y
    
    duration_gap = asset_duration - (liability_value / total_assets_base * liability_duration)
    # Standard Duration Gap = D_A - (L/A)*D_L
    
    print(f"Asset Duration: {asset_duration:.2f}")
    print(f"Liability Duration: {liability_duration:.2f}")
    print(f"L/A Ratio: {liability_value / total_assets_base:.2%}")
    print(f"Duration Gap: {duration_gap:.2f} years")
    
    # 4. Generate ML Stress Scenario (Forecast Dec 2025)
    print("\nGenerating ML Stress Scenario (Dec 2025 Forecast)...")
    
    model = ForecastingModel(lags=2)
    model.train(train_df)
    
    # Use One-Step Forecast to Dec 30, 2025 (Ex-Post Scenario)
    # This aligns with the "run_expected_vs_actual.py" logic.
    t1_date = pd.Timestamp('2025-12-30')
    
    # Needs full history to pick T1-1
    if t1_date not in yields_df.index:
         # Try to find nearest if exact date missing
         t1_date = yields_df.index[yields_df.index.get_indexer([t1_date], method='nearest')[0]]
    
    t1_loc = yields_df.index.get_loc(t1_date)
    context_data = yields_df.iloc[:t1_loc]
    
    final_forecast = model.predict_next_step(context_data)
    
    yields_shock = [final_forecast[f'tenor_{t}y'] for t in tenors]
    nss_shock = NSSModel()
    nss_shock.calibrate(tenors, yields_shock)
    
    # 5. Valuation under Stress
    # Note: In ALM Stress Test, we usually revalue at T0 using the Shocked Curve (Instantaneous Shock)
    # OR we project to T1. "Scenario" usually implies Instantaneous shift for Capital Requirement.
    # But here we use "Forecasted Dec 2025" as the "Scenario". 
    # Let's assume Instantaneous Shock to the Dec 2025 levels.
    
    p_10y_shock = bond_10y.calculate_price(nss_shock)
    p_50y_shock = bond_50y.calculate_price(nss_shock)
    total_assets_shock = p_10y_shock + p_50y_shock
    
    # Liability Shock (Approx via Duration)
    # Calculate avg yield shift
    avg_yield_base = np.mean(yields_t0)
    avg_yield_shock = np.mean(yields_shock)
    dy = (avg_yield_shock - avg_yield_base) / 100.0 # Percent to Decimal
    
    # Delta L = - L * D * dy
    delta_liab = - liability_value * liability_duration * dy
    liability_shock = liability_value + delta_liab
    
    # Results
    delta_assets = total_assets_shock - total_assets_base
    net_equity_base = total_assets_base - liability_value
    net_equity_shock = total_assets_shock - liability_shock
    delta_equity = net_equity_shock - net_equity_base
    
    print("\n--- ALM Stress Test Report (ML Scenario) ---")
    print(f"{'Metric':<20} | {'Base (T0)':<15} | {'Shocked (ML)':<15} | {'Change':<15}")
    print("-" * 75)
    print(f"{'Total Assets':<20} | {total_assets_base:,.2f}          | {total_assets_shock:,.2f}          | {delta_assets:+,.2f}")
    print(f"{'Total Liabilities':<20} | {liability_value:,.2f}          | {liability_shock:,.2f}          | {delta_liab:+,.2f}")
    print(f"{'Net Asset Value':<20} | {net_equity_base:,.2f}          | {net_equity_shock:,.2f}          | {delta_equity:+,.2f}")
    
    print("\n[Implication]")
    if delta_equity < 0:
        print("Net Asset Value DECLINED.")
        if duration_gap < 0:
            print("Negative Duration Gap amplified losses if rates fell (or Liability rose more).")
        else:
            print("Rising rates caused Assets to fall faster than Liabilities.")
            
        print("Conclusion: Increase capital buffer or adjust asset duration.")
    else:
        print("Net Asset Value INCREASED. The scenario was favorable.")

if __name__ == "__main__":
    run_stress_test()
