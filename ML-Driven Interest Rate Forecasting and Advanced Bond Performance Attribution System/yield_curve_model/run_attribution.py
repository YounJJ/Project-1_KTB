import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from models import NSSModel, BondPricing, AttributionModel

def run_attribution():
    loader = DataLoader()
    df = loader.load_market_data()
    
    # Define period: H2 2025 (Where Real Bonds exist)
    # Start: 2025-06-30
    # End:   2025-12-30
    start_date = pd.Timestamp('2025-06-30')
    end_date = pd.Timestamp('2025-12-30')
    
    if start_date not in df.index or end_date not in df.index:
        # Find nearest
        try:
            start_date = df.index[df.index.get_indexer([start_date], method='nearest')[0]]
            end_date = df.index[df.index.get_indexer([end_date], method='nearest')[0]]
        except IndexError:
             print("Error: Dates not found in market data.")
             return
        
    print(f"Attribution Period: {start_date.date()} to {end_date.date()}")
    
    # Load Real Bond Prices
    prices_10y_df = loader.load_bond_prices('10y', start_date=start_date, end_date=end_date)
    prices_50y_df = loader.load_bond_prices('50y', start_date=start_date, end_date=end_date)
    
    try:
        p_10y_start = prices_10y_df.loc[start_date, 'fair_value']
        p_10y_end = prices_10y_df.loc[end_date, 'fair_value']
        
        p_50y_start = prices_50y_df.loc[start_date, 'fair_value']
        p_50y_end = prices_50y_df.loc[end_date, 'fair_value']
    except KeyError:
        print("Error: Exact dates not found in Price Table. Using nearest or check data.")
        # Fallback simplistic
        p_10y_start = prices_10y_df['fair_value'].iloc[0]
        p_10y_end = prices_10y_df['fair_value'].iloc[-1]
        p_50y_start = prices_50y_df['fair_value'].iloc[0]
        p_50y_end = prices_50y_df['fair_value'].iloc[-1]

    
    tenors = [1, 3, 5, 10, 20, 30, 50]
    yields_start = df.loc[start_date, ['tenor_1y', 'tenor_3y', 'tenor_5y', 'tenor_10y', 'tenor_20y', 'tenor_30y', 'tenor_50y']].values
    yields_end = df.loc[end_date, ['tenor_1y', 'tenor_3y', 'tenor_5y', 'tenor_10y', 'tenor_20y', 'tenor_30y', 'tenor_50y']].values
    
    # 1. Calibrate Models
    nss_start = NSSModel()
    nss_start.calibrate(tenors, yields_start)
    
    nss_end = NSSModel()
    nss_end.calibrate(tenors, yields_end)
    
    # 2. Define Real Bonds (Matches User Specs)
    # 10Y: Issue 2025-06-10, Mat 2035-06-10, Cpn 2.625%
    # 50Y: Issue 2024-09-10, Mat 2074-09-10, Cpn 2.750%
    
    # 10Y Bond
    mat_date_10y = pd.Timestamp('2035-06-10')
    ttm_10y = (mat_date_10y - start_date).days / 365.0
    bond_10y = BondPricing(coupon_rate=0.02625, maturity_years=ttm_10y, face_value=10000)
    
    # 50Y Bond
    mat_date_50y = pd.Timestamp('2074-09-10')
    ttm_50y = (mat_date_50y - start_date).days / 365.0
    bond_50y = BondPricing(coupon_rate=0.0275, maturity_years=ttm_50y, face_value=10000)
    
    # 3. Attribution
    dt = (end_date - start_date).days / 365.0
    
    attr_model_10y = AttributionModel(bond_10y)
    res_10y = attr_model_10y.decompose(nss_start, nss_end, dt_years=dt, 
                                       market_price_start=p_10y_start, 
                                       market_price_end=p_10y_end)
    
    attr_model_50y = AttributionModel(bond_50y)
    res_50y = attr_model_50y.decompose(nss_start, nss_end, dt_years=dt,
                                       market_price_start=p_50y_start,
                                       market_price_end=p_50y_end)
    
    # Print Results
    def print_res(name, res):
        print(f"\n--- {name} Attribution ---")
        print(f"Start Price: {res['P_Start']:.2f}")
        print(f"End Price:   {res['P_End']:.2f}")
        print(f"Income:      {res['Income']:.2f}")
        print(f"Total P/L:   {res['Total_PL']:.2f}")
        print(f"  > Carry & Roll: {res['Carry_Rolldown']:.2f}")
        print(f"  > Rate Change:  {res['Rate_Change']:.2f}")
        print(f"  > Spread_Effect: {res['Spread_Effect']:.2f}")
        
    print_res("10Y Bond", res_10y)
    print_res("50Y Bond", res_50y)
    
    # 4. Waterfall Chart
    def plot_waterfall(name, res, ax):
        # Using Spread_Effect key
        components = ['Carry/Roll', 'Rate Change', 'Spread', 'Total P/L']
        values = [res['Carry_Rolldown'], res['Rate_Change'], res['Spread_Effect'], res['Total_PL']]
        
        # Calculate steps
        # Start at 0
        # Step 1: Carry (0 -> Carry)
        # Step 2: Rate (Carry -> Carry+Rate)
        # Step 3: Spread (Carry+Rate -> Carry+Rate+Spread = Total)
        # Step 4: Total (0 -> Total) - usually shown as full bar to compare
        
        # Using bottoms for stacked effect
        # Bar 1: Bottom=0
        # Bar 2: Bottom=Val1
        # Bar 3: Bottom=Val1+Val2
        # Bar 4: Bottom=0
        
        step_bottoms = [0, values[0], values[0] + values[1], 0]
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        colors[-1] = 'blue' # Total
        
        ax.bar(components, values, bottom=step_bottoms, color=colors)
        
        ax.set_title(f"{name} P/L Attribution")
        ax.grid(axis='y', alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_waterfall("10Y Bond (Issue Jun'25)", res_10y, axes[0])
    plot_waterfall("50Y Bond (Issue Sep'24)", res_50y, axes[1])
    
    plt.tight_layout()
    plt.savefig('attribution_waterfall.png')
    print("\nWaterfall chart saved to 'attribution_waterfall.png'")

if __name__ == "__main__":
    run_attribution()
