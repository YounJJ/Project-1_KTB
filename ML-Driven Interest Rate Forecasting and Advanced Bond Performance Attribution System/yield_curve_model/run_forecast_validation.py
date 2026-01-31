import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from models import ForecastingModel
from sklearn.metrics import mean_squared_error, r2_score

def run_forecast_validation():
    # Load Data
    loader = DataLoader()
    df = loader.load_market_data()
    
    # Define Test Period (H2 2025)
    test_start_date = '2025-07-01'
    df_test = df[df.index >= test_start_date]
    test_size = len(df_test)
    
    if test_size == 0:
        print("Error: No test data found for H2 2025.")
        return

    print(f"Total Data: {len(df)} rows")
    print(f"Test Data (H2 2025): {test_size} rows")

    # Initialize Model
    # Tenor-Specific models (No PCA), 2 lags
    model = ForecastingModel(lags=2)
    
    # Run Walk-Forward Validation
    # yields=df (full dataset), test_size=len(df_test)
    pred_df, actual_df = model.walk_forward_validation(df, test_size=test_size)
    
    # Calculate RMSE
    overall_rmse = np.sqrt(mean_squared_error(actual_df, pred_df))
    print(f"\nOverall Forecasting RMSE: {overall_rmse*100:.4f} bp")
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    tenors_to_plot = ['tenor_3y', 'tenor_10y', 'tenor_50y']
    
    for i, tenor in enumerate(tenors_to_plot):
        plt.subplot(3, 1, i+1)
        # Individual RMSE
        rmse = np.sqrt(mean_squared_error(actual_df[tenor], pred_df[tenor]))
        r2 = r2_score(actual_df[tenor], pred_df[tenor])
        print(f"{tenor} RMSE: {rmse*100:.4f} bp, R2: {r2:.4f}")

        plt.plot(actual_df.index, actual_df[tenor], label='Actual', color='black', linewidth=1.5)
        plt.plot(pred_df.index, pred_df[tenor], label=f'XGBoost Forecast (R²={r2:.2f})', color='blue', linestyle='--')
        plt.title(f"{tenor} Yield Forecast (H2 2025)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    import os
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig('images/forecast_validation.png')
    print("\nForecast validation plot saved to 'images/forecast_validation.png'")

if __name__ == "__main__":
    run_forecast_validation()
