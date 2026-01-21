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
    
    # Portfolio (Assume simple sum of the assets)
    p_10y_base = bond_10y.calculate_price(nss_base)
    p_50y_base = bond_50y.calculate_price(nss_base)
    total_assets_base = p_10y_base + p_50y_base
    
    print(f"Total Assets (Base): {total_assets_base:,.2f}")
    
    # 3. Liability (ALM Setting)
    # Ratio 80%
    liability_value = total_assets_base * 0.80
    
    # Liability Duration
    # Assume 15 years
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
    
    print("\n--- ML Shock Details ---")
    print(f"{'Tenor':<10} | {'Base Yield':<12} | {'Forecast':<12} | {'Shock (bp)':<12}")
    print("-" * 55)
    for i, tenor in enumerate(tenors):
        # yields_t0 is a 1D array (values from a Series)
        shock_bp = (yields_shock[i] - yields_t0[i]) * 100 
        print(f"{tenor}y{'':<8} | {yields_t0[i]:<12.3f} | {yields_shock[i]:<12.3f} | {shock_bp:+,.2f}")
    
    print("-" * 55)
    print(f"Average Shock (Parallel Equivalent): {(avg_yield_shock - avg_yield_base)*100:+.2f} bp")
    
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
        
    # --- Visualization: Stress Test Dashboard ---
    print("Generating Stress Test Dashboard...")
    import matplotlib.pyplot as plt
    
    # Data for Shock Details Table
    shock_data = []
    shock_columns = ["Tenor", "Base Yield", "Forecast", "Shock (bp)"]
    for i, tenor in enumerate(tenors):
        shock_bp = (yields_shock[i] - yields_t0[i]) * 100 
        shock_data.append([f"{tenor}y", f"{yields_t0[i]:.3f}", f"{yields_shock[i]:.3f}", f"{shock_bp:+.2f}"])
        
    shock_data.append(["Avg", "", "", f"{(avg_yield_shock - avg_yield_base)*100:+.2f}"])

    # Data for ALM Report Table
    alm_data = [
        ["Total Assets", f"{total_assets_base:,.2f}", f"{total_assets_shock:,.2f}", f"{delta_assets:+,.2f}"],
        ["Total Liabilities", f"{liability_value:,.2f}", f"{liability_shock:,.2f}", f"{delta_liab:+,.2f}"],
        ["Net Asset Value", f"{net_equity_base:,.2f}", f"{net_equity_shock:,.2f}", f"{delta_equity:+,.2f}"]
    ]
    alm_columns = ["Metric", "Base (T0)", "Shocked (ML)", "Change"]
    
    # Create Dashboard Figure
    fig = plt.figure(figsize=(14, 8))
    plt.suptitle("ALM Stress Test Dashboard (ML Scenario - Dec 2025)", fontsize=16, weight='bold')
    
    # Grid Layout: 2 Rows, 2 Columns
    # Row 1: Left (Shock Table), Right (ALM Table)
    # Row 2: Full Width (Bar Chart)
    # Added wspace=0.3 to create more space between the two tables in the first row
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], wspace=0.3)
    
    # 1. Shock Details Table (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title("ML Shock Details (Yield Curve Shift)", fontsize=12, weight='bold')
    table1 = ax1.table(cellText=shock_data, colLabels=shock_columns, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.3)
    
    # Style Table 1
    for (i, j), cell in table1.get_celld().items():
        if i == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif i == len(shock_data): # Average Row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')
    
    # 2. ALM Report Table (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title("ALM Stress Test Impact", fontsize=12, weight='bold')
    table2 = ax2.table(cellText=alm_data, colLabels=alm_columns, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)
    
    # Style Table 2
    for (i, j), cell in table2.get_celld().items():
        if i == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif j == 3: # Change Column color coding
            val_text = cell.get_text().get_text() # Get text string
            try:
                # Remove comma and + sign for float conversion
                val = float(val_text.replace(',', '').replace('+', ''))
                if val < 0:
                    cell.set_text_props(color='red')
                elif val > 0:
                    cell.set_text_props(color='green')
            except:
                pass

    # 3. Bar Chart (Bottom)
    ax3 = fig.add_subplot(gs[1, :])
    
    metrics = ["Total Assets", "Total Liabilities", "Net Asset Value"]
    base_vals = [total_assets_base, liability_value, net_equity_base]
    shock_vals = [total_assets_shock, liability_shock, net_equity_shock]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, base_vals, width, label='Base (T0)', color='#40466e')
    rects2 = ax3.bar(x + width/2, shock_vals, width, label='Shocked (ML)', color='#e06666')
    
    ax3.set_ylabel('Value (Currency Unit)')
    ax3.set_title('Financial Position comparison: Base vs Shocked')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f'{height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('stress_test_dashboard.png', dpi=300)
    print("Dashboard saved to 'stress_test_dashboard.png'")

if __name__ == "__main__":
    run_stress_test()
