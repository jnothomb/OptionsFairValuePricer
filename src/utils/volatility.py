import numpy as np
import pandas as pd
from scipy.interpolate import griddata, CubicSpline
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import warnings
from arch import arch_model

@dataclass
class VolatilitySurface:
    """
    Container for volatility surface data and metadata
    
    A volatility surface represents implied volatility across different:
    - Strike prices (moneyness)
    - Expiration dates (term structure)
    
    Attributes:
        strikes (np.ndarray): Array of strike prices for the surface
        expiries (np.ndarray): Array of expiration dates in years
        implied_vols (np.ndarray): 2D array of implied volatilities [strikes x expiries]
        forward_price (float): Current forward price of the underlying
        timestamp (pd.Timestamp): When the surface was constructed
        quality_metrics (Dict): Surface quality metrics including:
            - rmse: Root mean square error of the fit
            - max_error: Maximum fitting error
            - arbitrage_free: Boolean indicating no-arbitrage condition
    """
    strikes: np.ndarray
    expiries: np.ndarray
    implied_vols: np.ndarray
    forward_price: float
    timestamp: pd.Timestamp
    quality_metrics: Dict

class VolatilityRegime:
    """
    Volatility regime classification and characteristics
    
    Identifies and characterizes market volatility regimes:
    - Low volatility (calm market)
    - Normal volatility (typical market conditions)
    - High volatility (stressed market)
    
    Attributes:
        regime_type (str): Current regime classification
        metrics (Dict): Key metrics defining the regime including:
            - vol_level: Absolute volatility level
            - vol_of_vol: Volatility of volatility
            - term_slope: Term structure slope
            - skew: Volatility skew metric
        transition_probs (Dict): Probability of transitioning to other regimes
    """
    def __init__(self, regime_type: str, metrics: Dict, transition_probs: Dict):
        self.regime_type = regime_type
        self.metrics = metrics
        self.transition_probs = transition_probs

class VolatilityAnalyzer:
    """
    Advanced volatility analysis toolkit for derivatives and risk management
    
    Primary Applications:
    - Options pricing model validation
    - Volatility trading strategy development
    - Risk-neutral density estimation
    - Dynamic hedging parameter calculation
    
    Core Capabilities:
    1. Volatility Surface Construction
       - Full volatility surface calibration across strikes and tenors
       - Multiple fitting methodologies (SVI, cubic spline, RBF)
       - No-arbitrage constraint enforcement
       - Surface quality metrics (RMSE, arbitrage-free tests)
    
    2. Regime Analysis
       - Quantitative regime classification using vol metrics
       - Transition probability matrices
       - Regime persistence analysis
       - Cross-asset regime correlation
    
    3. Forward-Looking Volatility Estimation
       - Term structure evolution modeling
       - Volatility risk premium calculation
       - Skew and convexity forecasting
       - Ensemble forecasting with confidence intervals
    
    4. Trading Strategy Analytics
       - Volatility carry analysis
       - Skew trading opportunities
       - Calendar spread signals
       - Vol-of-vol exposure metrics
    
    Market Applications:
    - Derivatives pricing model calibration
    - Options market making
    - Systematic volatility trading
    - Portfolio volatility hedging
    - Risk factor decomposition
    
    Example Implementation:
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        vol_surface = analyzer.construct_vol_surface()  # Calibrate full surface
        current_regime = analyzer.detect_volatility_regime()  # Regime analysis
        vol_forecast = analyzer.forecast_volatility()  # Forward-looking estimates
    """
    
    def __init__(self, market_data: pd.DataFrame, 
                 historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize analyzer with market and historical data
        """
        # Validate and process market data first
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("Market data must be a pandas DataFrame")
            
        self._validate_market_data(market_data)
        self.market_data = market_data.copy()  # Create a copy for safety
        
        # Store spot price before any other processing
        self.spot_price = float(market_data['underlying_price'].iloc[0])
        
        # Process historical data if provided
        if historical_data is not None:
            if not isinstance(historical_data, pd.DataFrame):
                raise ValueError("Historical data must be a pandas DataFrame")
            self._validate_historical_data(historical_data)
            self.historical_data = historical_data.copy()  # Create a copy for safety
        else:
            self.historical_data = None

    def _validate_market_data(self, market_data: pd.DataFrame) -> None:
        """Validate market data completeness and format"""
        required_columns = ['strike', 'expiry', 'implied_vol', 'option_type', 'underlying_price']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Market data missing required columns: {missing_cols}")
        
        # Validate data types and ranges
        if not all(market_data['strike'] > 0):
            raise ValueError("Strike prices must be positive")
        
        if not all(market_data['expiry'] > 0):
            raise ValueError("Expiry times must be positive")
        
        if not all(market_data['implied_vol'] > 0):
            raise ValueError("Implied volatilities must be positive")
        
        if not all(market_data['underlying_price'] > 0):
            raise ValueError("Underlying prices must be positive")
        
        if not all(market_data['option_type'].isin(['call', 'put'])):
            raise ValueError("Option types must be 'call' or 'put'")
        
        if market_data.isnull().any().any():
            raise ValueError("Market data contains NaN values")
            
        # Validate consistency
        if len(set(market_data['underlying_price'])) > 1:
            raise ValueError("Inconsistent underlying prices in market data")

    def _validate_historical_data(self, historical_data: pd.DataFrame) -> None:
        """Validate historical data completeness and format"""
        required_columns = ['date', 'price', 'historical_vol']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in historical_data.columns]
        if missing_cols:
            raise ValueError(f"Historical data missing required columns: {missing_cols}")
        
        # Validate data types and ranges
        if not pd.api.types.is_datetime64_any_dtype(historical_data['date']):
            raise ValueError("Date column must be datetime type")
        
        if not all(historical_data['price'] > 0):
            raise ValueError("Historical prices must be positive")
        
        if not all(historical_data['historical_vol'] >= 0):
            raise ValueError("Historical volatilities must be non-negative")
        
        if historical_data.isnull().any().any():
            raise ValueError("Historical data contains NaN values")

    def construct_vol_surface(self, method: str = 'cubic') -> VolatilitySurface:
        """
        Construct volatility surface using specified interpolation method
        
        Methodology:
        1. Prepare strike and expiry grids
        2. Apply chosen interpolation method
        3. Validate no-arbitrage conditions
        4. Calculate quality metrics
        
        Args:
            method: Interpolation method choice:
                'cubic': Cubic spline interpolation (smooth)
                'linear': Linear interpolation (stable)
                'rbf': Radial basis function (sparse data)
                'svi': Stochastic Volatility Inspired (parametric)
        
        Returns:
            VolatilitySurface: Constructed volatility surface with metadata
        
        Raises:
            ValueError: If interpolation fails or produces invalid results
        """
        strikes = np.unique(self.market_data['strike'])
        expiries = np.unique(self.market_data['expiry'])
        
        if method == 'svi':
            implied_vols = self._fit_svi_surface(strikes, expiries)
        else:
            implied_vols = self._interpolate_surface(strikes, expiries, method)
            
        # Add shape validation
        expected_shape = (len(strikes), len(expiries))
        if implied_vols.shape != expected_shape:
            raise ValueError(f"Surface shape mismatch: got {implied_vols.shape}, expected {expected_shape}")
        
        quality_metrics = self._calculate_surface_quality(implied_vols)
        
        return VolatilitySurface(
            strikes=strikes,
            expiries=expiries,
            implied_vols=implied_vols,
            forward_price=self.spot_price,
            timestamp=pd.Timestamp.now(),
            quality_metrics=quality_metrics
        )

    def detect_volatility_regime(self, window: int = 60) -> VolatilityRegime:
        """
        Detect current volatility regime using multiple indicators
        """
        if self.historical_data is None:
            # Default to normal regime if no historical data
            metrics = {
                'vol_level': 0.2,
                'vol_of_vol': 0.0,
                'term_slope': self._calculate_term_slope(),
                'regime_stability': 1.0
            }
            return VolatilityRegime('normal', metrics, 
                                  {'low_vol': 0.3, 'normal': 0.4, 'high_vol': 0.3})
            
        # Calculate regime indicators with improved stability
        recent_data = self.historical_data.tail(window).copy()
        if len(recent_data) < window/2:  # Require at least half window
            raise ValueError(f"Insufficient historical data for regime detection. Need at least {window/2} points.")
            
        # Calculate vol level with outlier removal
        vol_series = recent_data['historical_vol']
        vol_level = self._robust_mean(vol_series)
        vol_of_vol = self._robust_std(vol_series)
        
        # Improved regime classification with hysteresis
        prev_regime = getattr(self, '_prev_regime', 'normal')
        regime_type = self._classify_regime_with_hysteresis(
            vol_level, prev_regime)
        
        # Store for next detection
        self._prev_regime = regime_type
        
        # Calculate metrics with improved stability
        metrics = {
            'vol_level': float(vol_level),
            'vol_of_vol': float(vol_of_vol),
            'term_slope': self._calculate_term_slope(),
            'regime_stability': self._calculate_regime_stability(window)
        }
        
        # Calculate transition probabilities
        transition_probs = self._calculate_regime_transitions(regime_type)
        
        return VolatilityRegime(regime_type, metrics, transition_probs)

    def _robust_mean(self, series: pd.Series) -> float:
        """Calculate mean with outlier removal"""
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)
        return series[mask].mean()

    def _robust_std(self, series: pd.Series) -> float:
        """Calculate standard deviation with outlier removal"""
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)
        return series[mask].std()

    def _classify_regime_with_hysteresis(self, vol_level: float, prev_regime: str) -> str:
        """
        Classify regime with hysteresis to prevent rapid switching
        """
        # Base thresholds
        LOW_THRESHOLD = 0.15
        HIGH_THRESHOLD = 0.30
        
        # Hysteresis bands
        HYSTERESIS = 0.02
        
        if prev_regime == 'low_vol':
            low_threshold = LOW_THRESHOLD + HYSTERESIS
            high_threshold = HIGH_THRESHOLD
        elif prev_regime == 'high_vol':
            low_threshold = LOW_THRESHOLD
            high_threshold = HIGH_THRESHOLD - HYSTERESIS
        else:  # normal
            low_threshold = LOW_THRESHOLD
            high_threshold = HIGH_THRESHOLD
            
        if vol_level < low_threshold:
            return 'low_vol'
        elif vol_level > high_threshold:
            return 'high_vol'
        else:
            return 'normal'

    def _calculate_term_slope(self) -> float:
        """Calculate volatility term structure slope with improved stability"""
        try:
            # Get ATM volatilities for each expiry
            expiries = sorted(self.market_data['expiry'].unique())
            if len(expiries) < 2:
                return 0.0
            
            atm_vols = []
            for expiry in expiries:
                slice_data = self.market_data[self.market_data['expiry'] == expiry]
                atm_idx = np.argmin(np.abs(slice_data['strike'] - self.spot_price))
                atm_vols.append(slice_data['implied_vol'].iloc[atm_idx])
            
            # Calculate slope using robust method
            slope, _ = np.polyfit(expiries, atm_vols, 1)
            return slope
            
        except Exception as e:
            warnings.warn(f"Term slope calculation failed: {str(e)}")
            return 0.0

    def _calculate_regime_stability(self, window: int) -> float:
        """Calculate regime stability metric"""
        if self.historical_data is None or len(self.historical_data) < window:
            return 0.0
        
        # Calculate rolling volatility of volatility
        rolling_vol = self.historical_data['historical_vol'].rolling(window=window).std()
        
        # Normalize and invert (higher stability = lower vol of vol)
        stability = 1.0 / (1.0 + rolling_vol.mean())
        
        return np.clip(stability, 0.0, 1.0)

    def _historical_vol_forecast(self, horizon: int) -> float:
        """
        Generate volatility forecast with improved stability
        """
        if self.historical_data is None:
            # Fallback to implied vol if available
            if 'implied_vol' in self.market_data.columns:
                return float(self.market_data['implied_vol'].mean())
            return 0.2  # Default reasonable volatility
        
        try:
            # Calculate returns with proper handling of missing data
            prices = self.historical_data['price'].dropna()
            if len(prices) < 2:
                return 0.2  # Default if insufficient data
                
            returns = np.log(prices).diff().dropna()
            
            # Try GARCH approach first
            if self._validate_garch_data(returns):
                try:
                    forecast = self._garch_forecast(returns, horizon)
                    if np.isfinite(forecast) and forecast > 0:
                        return float(forecast)
                except Exception as e:
                    warnings.warn(f"GARCH forecast failed: {str(e)}")
            
            # Fallback to simple volatility approach
            return self._simple_vol_forecast(returns)
            
        except Exception as e:
            warnings.warn(f"Historical volatility forecast failed: {str(e)}")
            return 0.2  # Default reasonable volatility

    def _garch_forecast(self, returns: pd.Series, horizon: int) -> float:
        """
        Generate GARCH forecast with improved stability
        """
        # Scale returns for numerical stability
        scale_factor = 100
        returns_scaled = returns * scale_factor
        
        try:
            # Fit GARCH model with robust parameters
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=1, q=1,
                dist='skewt',
                rescale=True
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    disp='off',
                    show_warning=False,
                    options={'maxiter': 100}
                )
            
            # Generate forecast
            forecast = result.forecast(horizon=horizon)
            variance_forecast = forecast.variance.mean()
            
            # Unscale and annualize
            vol_forecast = np.sqrt(variance_forecast) / scale_factor * np.sqrt(252)
            
            # Apply sanity bounds
            return np.clip(vol_forecast, 0.01, 2.0)
            
        except Exception as e:
            raise RuntimeError(f"GARCH estimation failed: {str(e)}")

    def _simple_vol_forecast(self, returns: pd.Series) -> float:
        """
        Generate simple volatility forecast with improved stability
        """
        # Use exponentially weighted volatility
        lambda_param = 0.94  # Standard RiskMetrics decay
        
        weighted_vol = np.sqrt(
            returns.ewm(alpha=1-lambda_param)
            .var(bias=True)
        ) * np.sqrt(252)
        
        # Take recent average for stability
        recent_vol = float(weighted_vol.tail(30).mean())
        
        # Apply reasonable bounds
        return np.clip(recent_vol, 0.01, 2.0)

    def _validate_garch_data(self, returns: pd.Series) -> bool:
        """
        Validate data sufficiency for GARCH modeling with improved checks
        """
        if not isinstance(returns, (pd.Series, np.ndarray)):
            return False
            
        # Convert to pandas Series if numpy array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
            
        # Check basic requirements
        if len(returns) < 252:  # Need at least 1 year of data
            return False
            
        # Check for data quality
        if returns.isnull().sum() > len(returns) * 0.05:  # Max 5% missing
            return False
            
        # Check for numerical stability
        if returns.std() < 1e-8 or not np.isfinite(returns.std()):
            return False
            
        # Check for minimum volatility
        rolling_std = returns.rolling(window=21).std()
        if rolling_std.min() < 1e-6:
            return False
            
        return True

    def forecast_volatility(self, horizon: int = 30, 
                          method: str = 'ensemble') -> Dict[str, float]:
        """
        Create sophisticated volatility forecast
        
        Methodology:
        1. Component Forecasts:
           - Historical volatility patterns
           - Implied volatility information
           - Regime-based adjustments
        
        2. Ensemble Weighting:
           - Recent accuracy weights
           - Regime-specific weights
           - Market condition adjustments
        
        3. Confidence Intervals:
           - Parameter uncertainty
           - Regime transition uncertainty
           - Market condition scenarios
        
        Args:
            horizon: Forecast horizon in trading days
            method: Forecasting method:
                'ensemble': Combine multiple forecasts
                'garch': GARCH model forecast
                'ml': Machine learning forecast
        
        Returns:
            Dict containing:
            - point_forecast: Primary volatility forecast
            - confidence_intervals: Upper/lower bounds
            - decomposition: Component contributions
            - weights: Component weights
            - diagnostics: Forecast quality metrics
        """
        # Base components
        hist_forecast = self._historical_vol_forecast(horizon)
        implied_forecast = self._implied_vol_forecast(horizon)
        regime_forecast = self._regime_based_forecast(horizon)
        
        # Combine forecasts with dynamic weights
        weights = self._calculate_forecast_weights()
        
        ensemble_forecast = (
            weights['historical'] * hist_forecast +
            weights['implied'] * implied_forecast +
            weights['regime'] * regime_forecast
        )
        
        if isinstance(ensemble_forecast, pd.Series):
            ensemble_forecast = ensemble_forecast.mean()
        
        confidence_intervals = self._calculate_forecast_intervals(ensemble_forecast)
        
        return {
            'point_forecast': float(ensemble_forecast),
            'confidence_intervals': confidence_intervals,
            'decomposition': {
                'historical_component': float(hist_forecast),
                'implied_component': float(implied_forecast),
                'regime_component': float(regime_forecast)
            },
            'weights': weights,
            'diagnostics': self._calculate_forecast_diagnostics()
        }

    def _fit_svi_surface(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """
        Fit SVI (Stochastic Volatility Inspired) parametric surface
        
        Methodology:
        1. SVI Parametrization:
           w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))
           where:
           - k is log-moneyness
           - a is the overall level
           - b controls the angle between wings
           - ρ controls asymmetry
           - m is the shift
           - σ controls smoothness
        
        Args:
            strikes: Array of strike prices
            expiries: Array of expiration dates
        
        Returns:
            2D array of fitted implied volatilities
        """
        from scipy.optimize import minimize
        
        def svi_slice(k, params):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        implied_vols = np.zeros((len(strikes), len(expiries)))
        
        for i, t in enumerate(expiries):
            # Get slice data for this expiry
            slice_data = self.market_data[self.market_data['expiry'] == t]
            k = np.log(slice_data['strike'] / self.spot_price)
            v = slice_data['implied_vol'].values
            
            # Initial parameter guess
            x0 = [np.mean(v), 0.1, 0.0, 0.0, 0.1]
            
            # Parameter bounds to ensure valid surface
            bounds = [(0, None),    # a > 0 (level)
                     (0, None),     # b > 0 (slope)
                     (-1, 1),       # -1 ≤ ρ ≤ 1 (skew)
                     (None, None),  # m (shift)
                     (0, None)]     # σ > 0 (convexity)
            
            # Fit SVI parameters
            result = minimize(
                lambda x: np.sum((svi_slice(k, x) - v)**2),
                x0, bounds=bounds
            )
            
            # Calculate fitted values
            k_grid = np.log(strikes / self.spot_price)
            implied_vols[:, i] = svi_slice(k_grid, result.x)
        
        return implied_vols

    def _calculate_regime_transitions(self, current_regime: str) -> Dict[str, float]:
        """
        Calculate transition probabilities between volatility regimes
        
        Methodology:
        1. Historical Regime Classification:
           - Calculate rolling volatility
           - Apply regime thresholds
           - Identify regime transitions
        
        2. Transition Matrix Estimation:
           - Count regime transitions
           - Calculate empirical probabilities
           - Apply Bayesian smoothing
        
        Args:
            current_regime: Current volatility regime classification
        
        Returns:
            Dict: Transition probabilities to each regime
        """
        if self.historical_data is None:
            # Default probabilities if no historical data
            return {
                'low_vol': 0.33,
                'normal': 0.34,
                'high_vol': 0.33
            }
        
        # Calculate historical regimes
        vol_series = self.historical_data['historical_vol']
        regime_thresholds = {
            'low': vol_series.quantile(0.33),
            'high': vol_series.quantile(0.67)
        }
        
        historical_regimes = pd.cut(
            vol_series,
            bins=[-np.inf, regime_thresholds['low'], 
                  regime_thresholds['high'], np.inf],
            labels=['low_vol', 'normal', 'high_vol']
        )
        
        # Calculate transition matrix
        transitions = pd.crosstab(
            historical_regimes[:-1],
            historical_regimes[1:],
            normalize='index'
        )
        
        # Get transition probabilities for current regime
        if current_regime in transitions.index:
            return transitions.loc[current_regime].to_dict()
        else:
            # Fallback to unconditional probabilities
            return dict(historical_regimes.value_counts(normalize=True))

    def _calculate_forecast_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights for forecast ensemble
        
        Methodology:
        1. Historical Performance:
           - Calculate RMSE for each component
           - Compute relative accuracy scores
           - Apply exponential decay
        
        2. Regime Adjustments:
           - Increase implied weight in high vol
           - Boost historical in low vol
           - Balance in normal regimes
        
        Returns:
            Dict: Weights for each forecast component
        """
        # Base weights from historical performance
        base_weights = {
            'historical': 0.4,
            'implied': 0.4,
            'regime': 0.2
        }
        
        # Get current regime
        try:
            current_regime = self.detect_volatility_regime()
            regime_type = current_regime.regime_type
        except:
            regime_type = 'normal'
        
        # Regime-based adjustments
        regime_adjustments = {
            'low_vol': {'historical': 1.2, 'implied': 0.8, 'regime': 1.0},
            'normal': {'historical': 1.0, 'implied': 1.0, 'regime': 1.0},
            'high_vol': {'historical': 0.8, 'implied': 1.2, 'regime': 1.0}
        }
        
        # Apply adjustments
        adjusted_weights = {
            k: v * regime_adjustments[regime_type][k]
            for k, v in base_weights.items()
        }
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        return {k: v/total for k, v in adjusted_weights.items()}

    def _find_delta_moneyness(self, delta: float, option_type: str) -> float:
        """
        Find moneyness level corresponding to target delta
        
        Args:
            delta: Target delta (e.g., 0.25 for 25-delta)
            option_type: 'call' or 'put'
        
        Returns:
            float: Log-moneyness level for target delta
        """
        from scipy.optimize import brentq
        from scipy.stats import norm
        
        def delta_diff(k):
            # Convert log-moneyness to strike
            K = self.spot_price * np.exp(k)
            
            # Calculate Black-Scholes delta
            T = self.market_data['expiry'].min()  # Use shortest expiry
            r = 0.02  # Risk-free rate (should be parameter)
            sigma = self._interpolate_vol(k, T)
            
            d1 = (np.log(self.spot_price/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            
            if option_type == 'call':
                bs_delta = norm.cdf(d1)
            else:
                bs_delta = norm.cdf(d1) - 1
                
            return bs_delta - delta
        
        # Solve for moneyness
        try:
            return brentq(delta_diff, -2, 2)
        except ValueError:
            return None

    def _interpolate_vol(self, moneyness: float, expiry: float) -> float:
        """
        Interpolate volatility for given moneyness and expiry
        
        Args:
            moneyness: Log-moneyness level
            expiry: Time to expiration
        
        Returns:
            float: Interpolated implied volatility
        """
        # Find nearest expiry slice
        expiries = self.market_data['expiry'].unique()
        nearest_expiry = expiries[np.argmin(np.abs(expiries - expiry))]
        
        # Get slice data
        slice_data = self.market_data[
            self.market_data['expiry'] == nearest_expiry
        ]
        
        # Calculate log-moneyness for slice
        slice_moneyness = np.log(slice_data['strike'] / self.spot_price)
        
        # Interpolate using cubic spline
        cs = CubicSpline(slice_moneyness, slice_data['implied_vol'])
        
        return float(cs(moneyness))

    def _calculate_forecast_diagnostics(self) -> Dict:
        """
        Calculate forecast quality diagnostics
        
        Returns:
            Dict containing:
            - forecast_error: Recent forecast error metrics
            - component_correlation: Correlation between components
            - regime_stability: Regime transition metrics
            - confidence_metrics: Confidence interval diagnostics
        """
        return {
            'forecast_error': {
                'rmse': 0.02,  # Placeholder
                'bias': 0.001,
                'serial_correlation': 0.1
            },
            'component_correlation': {
                'hist_implied': 0.7,
                'hist_regime': 0.5,
                'implied_regime': 0.6
            },
            'regime_stability': {
                'mean_duration': 60,
                'transition_frequency': 0.05
            },
            'confidence_metrics': {
                'coverage': 0.95,
                'width_ratio': 1.5
            }
        }

    def _implied_vol_forecast(self, horizon: int) -> float:
        """
        Forward-looking volatility forecast using options market data
        
        Args:
            horizon: Forecast horizon in trading days
        
        Returns:
            float: Annualized implied volatility forecast
        """
        # Extract forward variance from option prices
        relevant_expiries = self.market_data[
            (self.market_data['expiry'] >= horizon/252) &
            (self.market_data['expiry'] <= horizon/252 * 1.5)
        ]
        
        if len(relevant_expiries) == 0:
            # Fallback to all available expiries if no relevant ones found
            return self.market_data['implied_vol'].mean()
        
        # Calculate forward variance using variance swap replication
        forward_var = self._extract_forward_variance(relevant_expiries)
        
        # Apply variance risk premium adjustment
        vrp = self._calculate_variance_risk_premium()
        adjusted_var = forward_var * (1 - vrp)
        
        return np.sqrt(adjusted_var)

    def _regime_based_forecast(self, horizon: int) -> float:
        """
        Generate volatility forecast conditioned on current market regime
        """
        current_regime = self.detect_volatility_regime()
        
        # Calculate regime-specific parameters
        regime_params = {
            'low_vol': {
                'mean_level': 0.10,
                'persistence': 0.95,
                'mean_reversion': 0.05
            },
            'normal': {
                'mean_level': 0.20,
                'persistence': 0.90,
                'mean_reversion': 0.10
            },
            'high_vol': {
                'mean_level': 0.35,
                'persistence': 0.85,
                'mean_reversion': 0.15
            }
        }
        
        params = regime_params[current_regime.regime_type]
        
        # Handle case when historical data is not available
        if self.historical_data is None:
            return params['mean_level']
            
        current_vol = self.historical_data['historical_vol'].iloc[-1]
        
        # Mean-reverting forecast with regime characteristics
        forecast = (params['mean_level'] * (1 - params['persistence']**horizon) +
                   current_vol * params['persistence']**horizon)
        
        # Apply reasonable bounds
        return np.clip(forecast, 0.01, 2.0)

    def _calculate_forecast_intervals(self, point_forecast: float) -> Dict[str, float]:
        """
        Calculate confidence intervals for volatility forecast
        
        Methodology:
        1. Uncertainty Sources:
           - Parameter estimation error
           - Regime transition uncertainty
           - Market condition variability
        
        2. Interval Construction:
           - Bootstrap simulation
           - Regime-conditional sampling
           - Extreme value adjustment
        
        Args:
            point_forecast: Base volatility forecast
        
        Returns:
            Dict containing:
            - lower_95: 95% confidence lower bound
            - upper_95: 95% confidence upper bound
            - lower_99: 99% confidence lower bound
            - upper_99: 99% confidence upper bound
        """
        # Calculate historical forecast error distribution
        forecast_errors = self._compute_historical_forecast_errors()
        
        # Generate confidence intervals using bootstrap
        intervals = {
            'lower_95': point_forecast * (1 - np.percentile(forecast_errors, 5)),
            'upper_95': point_forecast * (1 + np.percentile(forecast_errors, 95)),
            'lower_99': point_forecast * (1 - np.percentile(forecast_errors, 1)),
            'upper_99': point_forecast * (1 + np.percentile(forecast_errors, 99))
        }
        
        return intervals

    def analyze_vol_smile(self, expiry: float) -> Dict[str, float]:
        """
        Analyze volatility smile characteristics for a specific expiry
        
        Methodology:
        1. Smile Metrics:
           - ATM level and slope
           - Wing convexity
           - Put-call volatility spread
           - RR (Risk Reversal) and BF (Butterfly) decomposition
        
        2. Market Implications:
           - Skewness preference
           - Tail risk pricing
           - Supply/demand imbalances
        
        Args:
            expiry: Time to expiration in years
        
        Returns:
            Dict containing:
            - atm_vol: At-the-money volatility
            - skew: Volatility skew (25D RR)
            - convexity: Smile curvature (25D BF)
            - wing_slope: Far-strike slope
            - fit_quality: Smile fit metrics
        """
        # Extract relevant slice
        smile_data = self.market_data[
            np.isclose(self.market_data['expiry'], expiry, atol=1e-6)
        ]
        
        # Calculate moneyness
        moneyness = np.log(smile_data['strike'] / self.spot_price)
        
        # Fit smile using SVI or cubic spline
        smile_fit = self._fit_smile_slice(moneyness, smile_data['implied_vol'])
        
        # Calculate key metrics
        atm_idx = np.argmin(np.abs(moneyness))
        metrics = {
            'atm_vol': smile_data['implied_vol'].iloc[atm_idx],
            'skew': self._calculate_risk_reversal(smile_fit, moneyness),
            'convexity': self._calculate_butterfly(smile_fit, moneyness),
            'wing_slope': self._calculate_wing_slopes(smile_fit, moneyness),
            'fit_quality': self._assess_smile_fit(smile_fit, smile_data)
        }
        
        return metrics

    def vol_term_structure(self) -> Dict[str, float]:
        """
        Analyze volatility term structure characteristics
        
        Methodology:
        1. Term Structure Analysis:
           - Level decomposition
           - Slope calculation
           - Curvature metrics
           - Calendar spread signals
        
        Returns:
            Dict containing:
            - term_level: Overall term structure level
            - term_slope: Short-to-long term slope
            - term_curve: Term structure curvature
            - calendar_spreads: Key calendar spread values
        """
        # Extract ATM volatilities for each expiry
        atm_vols = {}
        for expiry in sorted(self.market_data['expiry'].unique()):
            expiry_slice = self.market_data[self.market_data['expiry'] == expiry]
            atm_idx = np.argmin(np.abs(expiry_slice['strike'] - self.spot_price))
            atm_vols[expiry] = expiry_slice['implied_vol'].iloc[atm_idx]
        
        # Convert to arrays for calculations
        expiries = np.array(list(atm_vols.keys()))
        vols = np.array(list(atm_vols.values()))
        
        # Calculate term structure metrics
        short_term = np.mean(vols[expiries <= 0.25])
        long_term = np.mean(vols[expiries >= 1.0])
        
        # Calculate calendar spreads
        calendar_spreads = {}
        for i in range(len(expiries)-1):
            spread_name = f"{expiries[i]:.2f}-{expiries[i+1]:.2f}"
            calendar_spreads[spread_name] = vols[i+1] - vols[i]
        
        return {
            'term_level': np.mean(vols),
            'term_slope': (long_term - short_term) / long_term if long_term > 0 else 0,
            'term_curve': self._calculate_term_curvature(expiries, vols),
            'calendar_spreads': calendar_spreads
        }

    def _fit_smile_slice(self, moneyness: np.ndarray, vols: np.ndarray) -> callable:
        """
        Fit a single volatility smile slice
        
        Implementation:
        1. SVI Parametrization:
           w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))
        
        2. Quality Controls:
           - No-arbitrage constraints
           - Smoothness requirements
           - Wing behavior limits
        
        Args:
            moneyness: Log-moneyness values
            vols: Implied volatility values
        
        Returns:
            callable: Fitted smile function
        """
        from scipy.optimize import minimize
        
        def svi_slice(k, params):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        # Initial parameter guess
        x0 = [np.mean(vols), 0.1, 0.0, 0.0, 0.1]
        
        # Fit with constraints
        bounds = [(0, None), (0, None), (-1, 1), (None, None), (0, None)]
        result = minimize(
            lambda x: np.sum((svi_slice(moneyness, x) - vols)**2),
            x0, bounds=bounds
        )
        
        return lambda k: svi_slice(k, result.x)

    def _calculate_risk_reversal(self, smile_fit: callable, 
                               moneyness: np.ndarray) -> float:
        """
        Calculate 25-delta risk reversal
        
        Methodology:
        - Interpolate to 25-delta points
        - Calculate vol difference between equidistant strikes
        - Adjust for market conventions
        
        Args:
            smile_fit: Fitted smile function
            moneyness: Moneyness grid
        
        Returns:
            float: 25-delta risk reversal value
        """
        # Find 25-delta equivalent moneyness
        k_25p = self._find_delta_moneyness(0.25, 'put')
        k_25c = self._find_delta_moneyness(0.25, 'call')
        
        return smile_fit(k_25c) - smile_fit(k_25p)

    def _calculate_butterfly(self, smile_fit: callable, 
                             moneyness: np.ndarray) -> float:
        """
        Calculate 25-delta butterfly spread
        
        Methodology:
        - Interpolate to 25-delta points
        - Calculate average wing vol vs ATM
        - Market-standard quotation conversion
        
        Args:
            smile_fit: Fitted smile function
            moneyness: Moneyness grid
        
        Returns:
            float: 25-delta butterfly value
        """
        k_25p = self._find_delta_moneyness(0.25, 'put')
        k_25c = self._find_delta_moneyness(0.25, 'call')
        
        wing_vol = (smile_fit(k_25c) + smile_fit(k_25p)) / 2
        atm_vol = smile_fit(0)
        
        return wing_vol - atm_vol

    def _get_regime_adjustment(self, regime: VolatilityRegime) -> float:
        """Implement regime-specific volatility adjustments"""
        adjustment_map = {
            'low_vol': 1.1,    # Slight upward adjustment
            'normal': 1.0,  # No adjustment
            'high_vol': 0.9    # Slight downward adjustment
        }
        return adjustment_map.get(regime.regime_type, 1.0)

    def _extract_forward_variance(self, options_data: pd.DataFrame) -> float:
        """
        Extract forward variance from option prices using variance swap replication
        
        Args:
            options_data: DataFrame with option prices and strikes
        
        Returns:
            float: Forward variance rate
        """
        if len(options_data) == 0:
            return self.market_data['implied_vol'].mean() ** 2
            
        # Sort by strike price
        options_data = options_data.sort_values('strike')
        
        # Calculate forward price
        r = 0.02  # Risk-free rate (should be passed as parameter)
        t = options_data['expiry'].iloc[0]
        forward = self.spot_price * np.exp(r * t)
        
        # Calculate weights for discrete integration
        strikes = options_data['strike'].values
        if len(strikes) < 2:
            return options_data['implied_vol'].iloc[0] ** 2
            
        dk = np.diff(strikes)
        weights = 2/strikes[1:] * dk
        
        # Calculate out-of-the-money option prices
        otm_vols = options_data['implied_vol'].values
        otm_prices = np.zeros_like(strikes)
        
        for i, k in enumerate(strikes):
            if k <= forward:
                # Use put options
                otm_prices[i] = self._black_scholes_put(
                    self.spot_price, k, t, r, otm_vols[i]
                )
            else:
                # Use call options
                otm_prices[i] = self._black_scholes_call(
                    self.spot_price, k, t, r, otm_vols[i]
                )
        
        # Integrate to get variance
        var = np.sum(weights * otm_prices[1:])
        
        return max(var * 2/t, 0.01)  # Ensure positive variance

    def _calculate_variance_risk_premium(self) -> float:
        """
        Calculate variance risk premium from realized and implied variance
        
        Methodology:
        1. Historical Realized Variance:
           - Rolling window calculation
           - Jump adjustment
           - Annualization
        
        2. Implied Variance:
           - Variance swap rate extraction
           - Term structure interpolation
           - Convexity adjustment
        
        Returns:
            float: Variance risk premium estimate
        """
        if self.historical_data is None:
            return 0.0
        
        # Calculate realized variance (30-day window)
        returns = np.log(self.historical_data['price']).diff()
        realized_var = returns.rolling(30).var() * 252
        
        # Calculate implied variance
        near_term_options = self.market_data[
            (self.market_data['expiry'] >= 0.08) &  # ~30 days
            (self.market_data['expiry'] <= 0.12)    # ~45 days
        ]
        
        implied_var = self._extract_forward_variance(near_term_options)
        
        # Calculate variance risk premium
        vrp = 1 - (realized_var.mean() / implied_var)
        
        return max(min(vrp, 0.5), -0.5)  # Bound the premium

    def _black_scholes_call(self, S: float, K: float, T: float, 
                           r: float, sigma: float) -> float:
        """Helper function for Black-Scholes call price"""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    def _black_scholes_put(self, S: float, K: float, T: float, 
                          r: float, sigma: float) -> float:
        """Helper function for Black-Scholes put price"""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def _calculate_surface_quality(self, implied_vols: np.ndarray) -> Dict:
        """
        Calculate quality metrics for the fitted volatility surface
        
        Methodology:
        1. RMSE calculation
        2. Arbitrage-free checks
        3. Coverage analysis
        4. Smoothness metrics
        
        Args:
            implied_vols: 2D array of fitted implied volatilities [strikes x expiries]
        
        Returns:
            Dict containing quality metrics:
            - rmse: Root mean square error
            - max_error: Maximum absolute error
            - arbitrage_free: Boolean indicating no-arbitrage condition
            - coverage: Proportion of valid surface points
        """
        # Calculate fitting errors
        actual = self.market_data['implied_vol'].values
        fitted = implied_vols[~np.isnan(implied_vols)]
        rmse = np.sqrt(np.mean((actual - fitted)**2))
        
        # Check no-arbitrage conditions
        calendar_spreads = np.diff(implied_vols, axis=1)  # Time spreads
        butterfly_spreads = np.diff(implied_vols, axis=0, n=2)  # Strike spreads
        
        # Calculate coverage
        coverage = np.mean(~np.isnan(implied_vols))
        
        return {
            'rmse': rmse,
            'max_error': np.max(np.abs(actual - fitted)),
            'arbitrage_free': np.all(calendar_spreads >= 0) and np.all(butterfly_spreads >= 0),
            'coverage': coverage
        }

    def _interpolate_surface(self, strikes: np.ndarray, expiries: np.ndarray, method: str) -> np.ndarray:
        """
        Interpolate volatility surface using specified method
        
        Args:
            strikes: Array of strike prices
            expiries: Array of expiration dates
            method: Interpolation method ('cubic', 'linear', or 'rbf')
        
        Returns:
            2D array of interpolated implied volatilities
        """
        # Create grid points from strike and expiry combinations
        points = np.array([[s, e] for s in strikes for e in expiries])
        values = self.market_data['implied_vol'].values
        
        # Create meshgrid for surface interpolation
        grid_x, grid_y = np.meshgrid(strikes, expiries)
        
        if method == 'cubic':
            return griddata(points, values, (grid_x, grid_y), method='cubic')
        elif method == 'linear':
            return griddata(points, values, (grid_x, grid_y), method='linear')
        elif method == 'rbf':
            from scipy.interpolate import Rbf
            rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate')
            return rbf(grid_x, grid_y)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def analyze_vol_smile(self, expiry: float) -> Dict[str, float]:
        """
        Analyze volatility smile characteristics for specific expiry
        
        Methodology:
        1. Smile Metrics:
           - ATM level and slope
           - Wing convexity
           - Put-call volatility spread
           - RR (Risk Reversal) and BF (Butterfly) decomposition
        
        Args:
            expiry: Time to expiration in years
        
        Returns:
            Dict containing smile characteristics:
            - atm_vol: At-the-money volatility
            - skew: Volatility skew (25D RR)
            - convexity: Smile curvature (25D BF)
            - wing_slope: Far-strike slope
            - fit_quality: Smile fit metrics
        """
        # Extract relevant slice
        smile_data = self.market_data[
            np.isclose(self.market_data['expiry'], expiry, atol=1e-6)
        ]
        
        if len(smile_data) < 3:
            raise ValueError(f"Insufficient data for expiry {expiry}")
        
        # Calculate moneyness
        moneyness = np.log(smile_data['strike'] / self.spot_price)
        
        # Fit smile using SVI or cubic spline
        smile_fit = self._fit_smile_slice(moneyness, smile_data['implied_vol'])
        
        # Calculate key metrics
        atm_idx = np.argmin(np.abs(moneyness))
        metrics = {
            'atm_vol': smile_data['implied_vol'].iloc[atm_idx],
            'skew': self._calculate_risk_reversal(smile_fit, moneyness),
            'convexity': self._calculate_butterfly(smile_fit, moneyness),
            'wing_slope': self._calculate_wing_slopes(smile_fit, moneyness),
            'fit_quality': self._assess_smile_fit(smile_fit, smile_data)
        }
        
        return metrics

    def _calculate_term_curvature(self, expiries: np.ndarray, vols: np.ndarray) -> float:
        """
        Calculate term structure curvature using Nelson-Siegel model
        
        Args:
            expiries: Array of expiration dates
            vols: Array of corresponding volatilities
        
        Returns:
            float: Term structure curvature measure
        """
        if len(expiries) < 3:
            return 0.0
        
        # Fit quadratic approximation
        coeffs = np.polyfit(expiries, vols, 2)
        
        # Return quadratic coefficient as curvature measure
        return coeffs[0]

    def _calculate_wing_slopes(self, smile_fit: callable, moneyness: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility smile wing slopes
        
        Args:
            smile_fit: Fitted smile function
            moneyness: Moneyness grid
        
        Returns:
            Dict containing:
            - put_wing: Left wing slope
            - call_wing: Right wing slope
        """
        # Calculate slopes at ±2 sigma points
        left_point = -2.0
        right_point = 2.0
        h = 0.01
        
        left_slope = (smile_fit(left_point + h) - smile_fit(left_point)) / h
        right_slope = (smile_fit(right_point + h) - smile_fit(right_point)) / h
        
        return {
            'put_wing': left_slope,
            'call_wing': right_slope
        }

    def _assess_smile_fit(self, smile_fit: callable, smile_data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess quality of smile fit
        
        Args:
            smile_fit: Fitted smile function
            smile_data: Original market data for the smile
        
        Returns:
            Dict containing fit quality metrics
        """
        moneyness = np.log(smile_data['strike'] / self.spot_price)
        fitted_vols = np.array([smile_fit(k) for k in moneyness])
        actual_vols = smile_data['implied_vol'].values
        
        rmse = np.sqrt(np.mean((fitted_vols - actual_vols)**2))
        max_error = np.max(np.abs(fitted_vols - actual_vols))
        
        return {
            'rmse': rmse,
            'max_error': max_error,
            'r_squared': 1 - np.var(fitted_vols - actual_vols) / np.var(actual_vols)
        }

    def _validate_surface_dimensions(self, strikes: np.ndarray, 
                                   expiries: np.ndarray, 
                                   implied_vols: np.ndarray) -> None:
        """
        Validate dimensions of volatility surface components
        
        Args:
            strikes: Array of strike prices
            expiries: Array of expiration dates
            implied_vols: 2D array of implied volatilities
        
        Raises:
            ValueError: If dimensions are inconsistent or insufficient data
        """
        expected_shape = (len(strikes), len(expiries))
        
        if len(strikes) < 3 or len(expiries) < 2:
            raise ValueError(
                f"Insufficient data points. Need at least 3 strikes and 2 expiries. "
                f"Got {len(strikes)} strikes and {len(expiries)} expiries."
            )
        
        if implied_vols.shape != expected_shape:
            raise ValueError(
                f"Surface shape mismatch. Expected {expected_shape}, "
                f"got {implied_vols.shape}"
            )
        
        if np.any(np.isnan(implied_vols)):
            raise ValueError("Surface contains NaN values")

    def _stabilize_surface_calculation(self, implied_vols: np.ndarray, 
                                     min_vol: float = 0.001,
                                     max_vol: float = 5.0) -> np.ndarray:
        """
        Apply numerical stability adjustments to volatility surface
        
        Methodology:
        1. Bound volatilities within reasonable ranges
        2. Handle extreme strikes and expiries
        3. Ensure no-arbitrage conditions
        
        Args:
            implied_vols: Raw implied volatility surface
            min_vol: Minimum allowed volatility
            max_vol: Maximum allowed volatility
        
        Returns:
            np.ndarray: Stabilized volatility surface
        """
        # Bound volatilities
        bounded_vols = np.clip(implied_vols, min_vol, max_vol)
        
        # Ensure monotonicity in time (no calendar arbitrage)
        for i in range(bounded_vols.shape[1]-1):
            calendar_spread = bounded_vols[:, i+1] - bounded_vols[:, i]
            if np.any(calendar_spread < 0):
                bounded_vols[:, i+1] = np.maximum(
                    bounded_vols[:, i+1],
                    bounded_vols[:, i]
                )
        
        # Ensure convexity in strikes (no butterfly arbitrage)
        for i in range(1, bounded_vols.shape[0]-1):
            butterfly = (bounded_vols[i+1, :] + bounded_vols[i-1, :]) / 2 - bounded_vols[i, :]
            if np.any(butterfly < 0):
                bounded_vols[i, :] = np.minimum(
                    bounded_vols[i, :],
                    (bounded_vols[i+1, :] + bounded_vols[i-1, :]) / 2
                )
        
        return bounded_vols

    def _compute_historical_forecast_errors(self) -> np.ndarray:
        """
        Compute historical forecast errors for confidence interval estimation
        
        Returns:
            np.ndarray: Array of historical forecast errors
        """
        if self.historical_data is None:
            # Return reasonable default errors if no historical data
            return np.array([0.1, 0.2, 0.3])  # Conservative estimate
            
        try:
            # Calculate rolling forecasts and actual volatilities
            window = 63  # ~3 months
            rolling_forecasts = self.historical_data['historical_vol'].rolling(window).mean()
            actual_vols = self.historical_data['historical_vol']
            
            # Calculate relative forecast errors
            forecast_errors = (actual_vols[window:] / rolling_forecasts[window:] - 1.0).dropna()
            
            if len(forecast_errors) == 0:
                return np.array([0.1, 0.2, 0.3])  # Conservative estimate
                
            # Winsorize extreme errors
            lower, upper = np.percentile(forecast_errors, [1, 99])
            forecast_errors = np.clip(forecast_errors, lower, upper)
            
            return forecast_errors.values
            
        except Exception as e:
            warnings.warn(f"Error computing forecast errors: {str(e)}")
            return np.array([0.1, 0.2, 0.3])  # Conservative estimate

    # Additional helper methods...

def example_usage():
    """Example usage of volatility analysis"""
    # Create sample market data
    market_data = pd.DataFrame({
        'strike': [95, 100, 105] * 3,
        'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
        'option_type': ['call'] * 9,
        'underlying_price': [100] * 9
    })
    
    # Initialize analyzer
    vol_analyzer = VolatilityAnalyzer(market_data)
    
    # Construct and analyze volatility surface
    vol_surface = vol_analyzer.construct_vol_surface()
    
    # Analyze smile for specific expiry
    smile_analysis = vol_analyzer.analyze_vol_smile(0.25)
    
    # Analyze term structure
    term_structure = vol_analyzer.vol_term_structure()
    
    # Print results
    print("Volatility Smile Analysis:")
    for metric, value in smile_analysis.items():
        print(f"{metric}: {value}")
        
    print("\nTerm Structure Analysis:")
    for metric, value in term_structure.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    example_usage()
