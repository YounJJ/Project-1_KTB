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
    
    # Attribution calculations
    dt = (end_date - start_date).days / 365.0
    
    attr_model_10y = AttributionModel(bond_10y)
    res_10y = attr_model_10y.decompose(nss_start, nss_end, dt_years=dt, 
                                       market_price_start=p_10y_start, 
                                       market_price_end=p_10y_end)
    res_10y['Bond'] = '10Y KTB'
    
    attr_model_50y = AttributionModel(bond_50y)
    res_50y = attr_model_50y.decompose(nss_start, nss_end, dt_years=dt,
                                       market_price_start=p_50y_start,
                                       market_price_end=p_50y_end)
    res_50y['Bond'] = '50Y KTB'

    results_list = [res_10y, res_50y]
    df_results = pd.DataFrame(results_list)
    
    # Reorder columns for better table appearance
    cols = ['Bond', 'P_Start', 'P_End', 'Income', 'Total_PL', 'Carry_Rolldown', 'Rate_Change', 'Spread_Effect']
    df_results = df_results[cols]
    df_results.columns = ['Bond', 'Start Price', 'End Price', 'Income', 'Total P/L', 'Carry/Roll', 'Rate Change', 'Spread Effect']

    # 1. Print Pretty Table in Terminal
    print("\n--- Attribution Summary Table ---")
    print(df_results.to_string(index=False))
    
    # 2. Save Summary Table as PNG
    def save_results_table(df, filename='attribution_table.png'):
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.axis('off')
        table = ax.table(cellText=df.round(2).values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            else:
                cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to '{filename}'")

    save_results_table(df_results)

    # 4. Waterfall Chart
    def plot_waterfall(name, res, ax):
        components = ['Carry/Roll', 'Rate Change', 'Spread', 'Total P/L']
        values = [res['Carry_Rolldown'], res['Rate_Change'], res['Spread_Effect'], res['Total_PL']]
        
        step_bottoms = [0, values[0], values[0] + values[1], 0]
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
        colors[-1] = '#3498db' # Total
        
        bars = ax.bar(components, values, bottom=step_bottoms, color=colors, edgecolor='black', alpha=0.8)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            bottom = bar.get_x() + bar.get_width() / 2
            val = values[i]
            y_pos = step_bottoms[i] + height/2 if i < 3 else height/2
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:+.1f}', 
                    ha='center', va='center', fontweight='bold', color='white' if abs(val) > 10 else 'black')

        ax.set_title(f"{name} P/L Attribution", fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel("P/L Amount")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_waterfall("10Y Bond (Issue Jun'25)", res_10y, axes[0])
    plot_waterfall("50Y Bond (Issue Sep'24)", res_50y, axes[1])
    
    plt.suptitle("Bond Performance Attribution Decomposition", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('attribution_waterfall.png', dpi=300, bbox_inches='tight')
    print("Waterfall chart saved to 'attribution_waterfall.png'")

if __name__ == "__main__":
    run_attribution()
