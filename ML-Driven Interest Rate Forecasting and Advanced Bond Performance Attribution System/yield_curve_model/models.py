import numpy as np
import pandas as pd

class NSSModel:
    def __init__(self):
        self.beta0 = 0.0
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.beta3 = 0.0
        self.tau1 = 1.0
        self.tau2 = 1.0
        self.is_fitted = False

    def _nss_factor_loading(self, t, tau1, tau2):
        """Calculates the factor loadings for Beta1, Beta2, Beta3."""
        # Avoid division by zero
        tau1 = np.maximum(tau1, 1e-4)
        tau2 = np.maximum(tau2, 1e-4)

        term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
        term2 = term1 - np.exp(-t / tau1)
        term3 = ((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2)
        
        return term1, term2, term3

    def calibrate(self, maturities, yields):
        """
        Calibrates the NSS model using Fixed-Tau Method (Grid Search + OLS).
        maturities: array-like, years to maturity (e.g., [1, 3, 5, ...])
        yields: array-like, observed yields corresponding to maturities
        """
        maturities = np.array(maturities)
        yields = np.array(yields)

        # Coarse Grid Search for Tau1, Tau2
        # range: 0.5 to 10.0, step 0.5
        tau_grid = np.arange(0.5, 10.5, 0.5)
        
        best_rmse = np.inf
        best_params = None

        # We'll use a loop for clarity and simplicity first.
        
        for t1 in tau_grid:
            for t2 in tau_grid:
                if t1 == t2:
                    continue # Avoid collinearity if possible, though NSS allows t1=t2 theoretically, 
                             # numerically it might be unstable or redundant. We skip for robustness.

                # Construct Regressor Matrix X
                # NSS: y(t) = beta0 + beta1 * L1 + beta2 * L2 + beta3 * L3
                L1, L2, L3 = self._nss_factor_loading(maturities, t1, t2)
                
                # Stack column vectors: [1, L1, L2, L3]
                ones = np.ones_like(maturities)
                X = np.column_stack((ones, L1, L2, L3))
                
                # OLS: Solve X * beta = yields
                # beta, residuals, rank, s = np.linalg.lstsq(X, yields, rcond=None)
                # Using lstsq to minimize squared error
                beta, residuals, _, _ = np.linalg.lstsq(X, yields, rcond=None)
                
                if residuals.size > 0:
                    rss = residuals[0]
                    rmse = np.sqrt(rss / len(yields))
                else:
                    # Perfect fit or calculation oddity
                    fitted = X @ beta
                    rss = np.sum((yields - fitted)**2)
                    rmse = np.sqrt(rss / len(yields))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'beta0': beta[0],
                        'beta1': beta[1],
                        'beta2': beta[2],
                        'beta3': beta[3],
                        'tau1': t1,
                        'tau2': t2
                    }

        if best_params:
            self.beta0 = best_params['beta0']
            self.beta1 = best_params['beta1']
            self.beta2 = best_params['beta2']
            self.beta3 = best_params['beta3']
            self.tau1 = best_params['tau1']
            self.tau2 = best_params['tau2']
            self.is_fitted = True
            return best_rmse
        else:
            raise ValueError("NSS Calibration failed to find optimal parameters.")

    def get_spot_rate(self, t):
        """Calculates spot rate for maturity t."""
        if not self.is_fitted:
            raise RuntimeError("Model is not calibrated.")
        
        t = np.array(t)
        
        # Vectorized safe calculation
        t_safe = np.maximum(t, 1e-6)
        
        term1, term2, term3 = self._nss_factor_loading(t_safe, self.tau1, self.tau2)
        
        r = self.beta0 + self.beta1 * term1 + self.beta2 * term2 + self.beta3 * term3
        return r

    def get_discount_factor(self, t):
        """Calculates discount factor Z(t) = exp(-r(t) * t)."""
        r = self.get_spot_rate(t)
        # r is usually in percent (e.g. 3.5), so divide by 100 if input yielded in percent.
         
        return np.exp(-(r / 100.0) * t)

class ForecastingModel:
    def __init__(self, lags=2):
        self.lags = lags
        self.models = {} # Dictionary of models per tenor
        self.is_fitted = False
        self.tenors = [] # List of column names forecasted

    def prepare_single_tenor_features(self, series):
        """
        Creates differenced features and lags for a single tenor series.
        series: pd.Series of yields
        """
        # 1. Differencing (Stationarity)
        diff = series.diff().dropna()
        
        X = []
        y = []
        indices = diff.index[self.lags:]
        
        # Create lag features
        for i in range(self.lags, len(diff)):
            # Features: Previous 'lags' changes
            lag_window = diff.iloc[i-self.lags:i].values
            target = diff.iloc[i]
            X.append(lag_window)
            y.append(target)
            
        return np.array(X), np.array(y), indices

    def train(self, yields_df):
        """
        Trains independent XGBoost models for each tenor column.
        yields_df: DataFrame (T x N_tenors)
        """
        from xgboost import XGBRegressor
        
        self.tenors = yields_df.columns.tolist()
        self.models = {}
        
        for tenor in self.tenors:
            # print(f"Training model for {tenor}...")
            X, y, _ = self.prepare_single_tenor_features(yields_df[tenor])
            
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, objective='reg:squarederror')
            model.fit(X, y)
            self.models[tenor] = model
            
        self.is_fitted = True

    def predict_next_step(self, current_history_df):
        """
        Predicts next step yields for all tenors.
        current_history_df: DataFrame containing enough history for lags.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
            
        pred_next = {}
        
        for tenor in self.tenors:
            series = current_history_df[tenor]
            
            # 1. Get recent diffs
            recent_diff = series.iloc[-(self.lags+1):].diff().dropna()
            
            if len(recent_diff) < self.lags:
                raise ValueError(f"Not enough history for {tenor}")
                
            # 2. Form input (lags)
            input_vec = recent_diff.iloc[-self.lags:].values.reshape(1, -1)
            
            # 3. Predict Delta
            pred_delta = self.models[tenor].predict(input_vec)[0]
            
            # 4. Reconstruct Level
            last_level = series.iloc[-1]
            pred_level = last_level + pred_delta
            pred_next[tenor] = pred_level
            
        return pd.Series(pred_next)

    def walk_forward_validation(self, yields_df, test_size=126):
        """
        Performs recursive walk-forward validation (One-step ahead).
        """
        # Split
        train_data = yields_df.iloc[:-test_size]
        test_data = yields_df.iloc[-test_size:]
        
        # Train on initial history
        self.train(train_data)
        
        predictions = []
        actuals = []
        dates = []
        
        # We need full history accessible to look back
        full_history = yields_df.copy()
        
        print(f"Starting Tenor-Specific Walk-Forward Validation on {len(test_data)} steps...")
        
        for i in range(len(test_data)):
            # Index where test step i is located in full_history
            current_idx = len(train_data) + i
            
            # Define context window (up to t-1)
            # We need enough rows to calculate 'lags' differences.
            # Lags=2 -> need 3 previous levels (to get 2 diffs).
            context_start = current_idx - (self.lags + 2) 
            if context_start < 0: context_start = 0
            
            context = full_history.iloc[context_start:current_idx]
            
            # Predict t
            pred_series = self.predict_next_step(context)
            
            predictions.append(pred_series)
            actuals.append(test_data.iloc[i])
            dates.append(test_data.index[i])
            
        return pd.DataFrame(predictions, index=dates), pd.DataFrame(actuals, index=dates)


class BondPricing:
    def __init__(self, coupon_rate, maturity_years, face_value=10000, frequency=2):
        """
        std coupon bond.
        coupon_rate: annual coupon rate (decimal, e.g. 0.03 for 3%)
        maturity_years: years until maturity
        face_value: par value
        frequency: coupon frequency per year (2 for semi-annual)
        """
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.face_value = face_value
        self.frequency = frequency
        
    def _generate_cashflows(self):
        """Generates (time, amount) pairs for all cashflows."""
        # Assume issued today for simplicity, or just calculate from now until M.
        # Times: 0.5, 1.0, 1.5, ... M
        n_periods = int(self.maturity_years * self.frequency)
        times = np.arange(1, n_periods + 1) / self.frequency
        
        coupon_amount = (self.coupon_rate * self.face_value) / self.frequency
        cashflows = np.full(n_periods, coupon_amount)
        cashflows[-1] += self.face_value # Add principal at maturity
        
        return times, cashflows

    def calculate_price(self, nss_model):
        """
        Calculates Fair Price using DCF and NSS spot rates.
        nss_model: calibrated NSSModel instance
        """
        times, cashflows = self._generate_cashflows()
        
        # Get spot rates for each cashflow time
        spot_rates = nss_model.get_spot_rate(times) # Returns percentage (e.g. 3.5)
        
        # Discount Factors: exp(-r*t)
        # Note: NSS get_spot_rate returns rate in %, so divide by 100
        discount_factors = np.exp(-(spot_rates / 100.0) * times)
        
        price = np.sum(cashflows * discount_factors)
        return price

    def calculate_macaulay_duration(self, nss_model):
        """Calculates Macaulay Duration (in years)."""
        times, cashflows = self._generate_cashflows()
        spot_rates = nss_model.get_spot_rate(times)
        discount_factors = np.exp(-(spot_rates / 100.0) * times)
        
        pv_cashflows = cashflows * discount_factors
        price = np.sum(pv_cashflows)
        
        if price == 0: return 0
        
        weighted_time = np.sum(times * pv_cashflows)
        return weighted_time / price

    def calculate_effective_duration(self, nss_model, shock_bp=10):
        """
        Calculates Effective Duration using parallel shift approximation.
        EffDur = (P_down - P_up) / (2 * P0 * dy)
        shock_bp: parallel shift in basis points (e.g. 10bp = 0.001)
        """
        # We need to shift the CURVE, not just the single bond yield.
        # But NSS factors describe the curve.
        # A parallel shift means shifting Beta0 (Level) by +/- shock
        
        # Base Price
        p0 = self.calculate_price(nss_model)
        
        # Shift Up
        original_beta0 = nss_model.beta0
        
        # P_up: Yields go UP (Price goes DOWN)
        nss_model.beta0 = original_beta0 + (shock_bp / 100.0) # Rate is %, so add bp/100? No. 
        # shock_bp=10 -> 0.1%. If rate is 3%, becomes 3.1%. Correct.
        p_up = self.calculate_price(nss_model)
        
        # P_down: Yields go DOWN (Price goes UP)
        nss_model.beta0 = original_beta0 - (shock_bp / 100.0)
        p_down = self.calculate_price(nss_model)
        
        # Restore model
        nss_model.beta0 = original_beta0
        
        dy = (shock_bp / 10000.0) # Change in DECIMAL yield (e.g. 10bp = 0.001)
        # Actually standard formula assumes dy is decimal change.
        # Here we shifted beta0 by 0.1 (since beta0 is %). 
        # So the yield curve shifted by 0.1%. 
        # The formula usually takes dy as decimal change (0.001).
        
        eff_dur = (p_down - p_up) / (2 * p0 * (shock_bp / 10000.0))
        return eff_dur

class AttributionModel:
    def __init__(self, bond):
        """
        bond: BondPricing instance
        """
        self.bond = bond

    def decompose(self, nss_start, nss_end, dt_years=0.5, market_price_start=None, market_price_end=None):
        """
        Decomposes P/L between T and T+dt.
        
        Logic for End-of-Period Pricing:
        We must account for the bond aging (Roll-down).
        At time T+dt, the bond has reduced maturity = Original Maturity - dt.
        We create a temporary bond instance to represent this 'aged' bond.
        """
        # 1. Prices (Model)
        # P_start: Price at T using Curve T and FULL Maturity
        p_start_model = self.bond.calculate_price(nss_start)
        
        # Define 'Aged Bond' (Maturity - dt)
        original_maturity = self.bond.maturity_years
        remaining_maturity = original_maturity - dt_years
        
        # Create temporary bond for T+dt calculations
        bond_at_end = BondPricing(
            coupon_rate=self.bond.coupon_rate,
            maturity_years=remaining_maturity,
            face_value=self.bond.face_value,
            frequency=self.bond.frequency
        )
        
        # P_end: Price at T+dt using Curve T+dt and REDUCED Maturity
        p_end_model = bond_at_end.calculate_price(nss_end)
        
        # 2. Income (Coupon Accrual)
        income = (self.bond.coupon_rate * self.bond.face_value) * dt_years
        
        # 3. Attribution Components
        
        # A. Carry & Roll-down (Time Effect)
        # Price at T+dt assuming Curve T (unchanged curve) but Time passed (Reduced Maturity)
        # P_carry = P(Maturity - dt, Curve_Start)
        p_carry = bond_at_end.calculate_price(nss_start)
        
        # Carry = (P_carry - P_start) + Income
        carry_rolldown = (p_carry - p_start_model) + income 
        
        # B. Rate Change (Curve Effect)
        # P(Maturity - dt, Curve_End) - P(Maturity - dt, Curve_Start)
        # equals p_end_model - p_carry
        rate_change = p_end_model - p_carry
        
        # Total Model P/L
        total_model_pl = carry_rolldown + rate_change
        
        # 4. Spread / Residual Calculation
        if market_price_start is not None and market_price_end is not None:
            # Total Actual P/L
            total_actual_pl = (market_price_end - market_price_start) + income
            
            # Spread Effect = Actual - Model
            spread_effect = total_actual_pl - total_model_pl
            
            # Use Actual as Total
            total_pl = total_actual_pl
            p_start_report = market_price_start
            p_end_report = market_price_end
        else:
            # Fallback to Model
            spread_effect = 0.0
            total_pl = total_model_pl
            p_start_report = p_start_model
            p_end_report = p_end_model
        
        return {
            "Total_PL": total_pl,
            "Income": income,
            "Carry_Rolldown": carry_rolldown,
            "Rate_Change": rate_change,
            "Spread_Effect": spread_effect,
            "P_Start": p_start_report,
            "P_End": p_end_report
        }



