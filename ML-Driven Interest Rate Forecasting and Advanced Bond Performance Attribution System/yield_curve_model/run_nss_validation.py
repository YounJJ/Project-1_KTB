import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from models import NSSModel

def run_validation():
    # Load Data
    loader = DataLoader() # Uses default yield_curve.db
    df = loader.load_market_data()
    
    if df.empty:
        print("Error: No data found in database.")
        return

    # Tenors in years
    tenors = [1, 3, 5, 10, 20, 30, 50]
    tenor_cols = ['tenor_1y', 'tenor_3y', 'tenor_5y', 'tenor_10y', 'tenor_20y', 'tenor_30y', 'tenor_50y']

    # Select sample dates
    # First day, Middle day, Last available day
    sample_dates = [
        df.index[0],
        df.index[len(df)//2],
        df.index[-1]
    ]
    
    model = NSSModel()
    
    plt.figure(figsize=(12, 8))
    
    print(f"{'Date':<12} | {'RMSE (bp)':<10} | {'Beta0':<7} | {'Beta1':<7} | {'Beta2':<7} | {'Beta3':<7} | {'Tau1':<5} | {'Tau2':<5}")
    print("-" * 90)

    for i, date in enumerate(sample_dates):
        # Get yields for the date
        yields = df.loc[date, tenor_cols].values
        
        try:
            # Calibrate
            rmse = model.calibrate(tenors, yields)
            
            # Print Stats
            print(f"{date.strftime('%Y-%m-%d'):<12} | {rmse*100:.4f}     | {model.beta0:.3f}   | {model.beta1:.3f}   | {model.beta2:.3f}   | {model.beta3:.3f}   | {model.tau1:.2f}  | {model.tau2:.2f}")
            
            # Plot
            # Generate smooth curve
            t_smooth = np.linspace(0.1, 50, 100)
            y_smooth = model.get_spot_rate(t_smooth)
            
            plt.subplot(2, 2, i+1)
            plt.scatter(tenors, yields, color='red', label='Actual Yields')
            plt.plot(t_smooth, y_smooth, label=f'NSS Fit (RMSE={rmse*100:.2f}bp)')
            plt.title(f"Yield Curve: {date.strftime('%Y-%m-%d')}")
            plt.xlabel("Maturity (Years)")
            plt.ylabel("Yield (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error fitting {date}: {e}")

    plt.tight_layout()
    plt.savefig('nss_validation.png')
    print("\nValidation plot saved to 'nss_validation.png'")

if __name__ == "__main__":
    run_validation()
