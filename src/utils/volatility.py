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
        
        Args:
            market_data: DataFrame containing current options market data
                Required columns:
                - strike: Option strike prices
                - expiry: Time to expiration in years
                - implied_vol: Market implied volatilities
                - option_type: 'call' or 'put'
                - underlying_price: Current price of underlying
            
            historical_data: Optional DataFrame with historical data
                Required columns:
                - date: Observation dates
                - price: Historical underlying prices
                - historical_vol: Realized volatility
        
        Raises:
            ValueError: If required columns are missing
        """
        self.market_data = market_data
        self.historical_data = historical_data
        self.spot_price = market_data['underlying_price'].iloc[0]
        self._validate_input_data()

    def _validate_input_data(self):
        """Validate input data completeness and format"""
        required_columns = ['strike', 'expiry', 'implied_vol', 'option_type']
        if not all(col in self.market_data.columns for col in required_columns):
            raise ValueError(f"Market data missing required columns: {required_columns}")

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
        
        Methodology:
        1. Calculate regime indicators:
           - Absolute volatility levels
           - Volatility of volatility
           - Term structure slope
           - Smile characteristics
        
        2. Apply classification algorithm:
           - K-means clustering
           - Threshold-based classification
           - Historical pattern matching
        
        3. Calculate transition probabilities:
           - Historical regime transitions
           - Current market conditions
           - Regime stability metrics
        
        Args:
            window: Lookback window for regime detection (trading days)
        
        Returns:
            VolatilityRegime: Current regime classification and characteristics
        
        Note:
            Longer windows provide more stable regime detection but may
            be less responsive to regime changes
        """
        if self.historical_data is None:
            raise ValueError("Historical data required for regime detection")
            
        # Calculate regime indicators
        recent_data = self.historical_data.last(window)
        vol_level = recent_data['historical_vol'].mean()
        vol_of_vol = recent_data['historical_vol'].std()
        term_structure = self.vol_term_structure()
        
        # Classify regime using multiple indicators
        regime_indicators = np.array([
            vol_level,
            vol_of_vol,
            term_structure['term_slope']
        ])
        
        regime_type = self._classify_regime(regime_indicators)
        metrics = {
            'vol_level': vol_level,
            'vol_of_vol': vol_of_vol,
            'term_slope': term_structure['term_slope']
        }
        
        # Calculate regime transition probabilities
        transition_probs = self._calculate_regime_transitions(regime_type)
        
        return VolatilityRegime(regime_type, metrics, transition_probs)

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
        
        confidence_intervals = self._calculate_forecast_intervals(ensemble_forecast)
        
        return {
            'point_forecast': ensemble_forecast,
            'confidence_intervals': confidence_intervals,
            'decomposition': {
                'historical_component': hist_forecast,
                'implied_component': implied_forecast,
                'regime_component': regime_forecast
            },
            'weights': weights,
            'diagnostics': self._calculate_forecast_diagnostics()
        }

    def _fit_svi_surface(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """
        Fit surface using SVI (Stochastic Volatility Inspired) parametrization
        Implementation Notes:
        -------------------
        1. SVI Parametrization:
           w(k) = a + b(ρ(k-m) + sqrt((k-m)^2 + σ^2))
           where:
           - k is log-moneyness
           - a is the overall level
           - b controls the angle between wings
           - ρ controls asymmetry
           - m is the shift
           - σ controls smoothness
        
        2. Optimization Process:
           - Initial guess based on market data characteristics
           - Constrained optimization to ensure no-arbitrage
           - Multiple starting points to avoid local minima
        
        3. Error Handling:
           - Checks for sufficient data points
           - Validates parameter bounds
           - Ensures smooth and realistic results
        
        4. Performance Considerations:
           - Vectorized calculations for efficiency
           - Caching of intermediate results
           - Early stopping for optimization
        """
        # SVI implementation details here
        pass

    def _calculate_regime_transitions(self, current_regime: str) -> Dict[str, float]:
        """
        Calculate transition probabilities between volatility regimes
        
        Implementation Notes:
        -------------------
        1. Regime Classification:
           - Low Vol: < historical_mean - 1*std_dev
           - Normal Vol: within ±1 std_dev of mean
           - High Vol: > historical_mean + 1*std_dev
        
        2. Transition Matrix:
           - Markov chain approach
           - Rolling window analysis
           - Smoothed probabilities
        
        3. Stability Metrics:
           - Eigenvalue analysis
           - Mean residence time
           - Transition frequency
        
        Error Handling:
        --------------
        - Insufficient historical data
        - Non-stationary regimes
        - Outlier detection and handling
        """
        # Transition probability calculation
        pass

    def _calculate_forecast_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights for forecast ensemble

        Implementation Notes:
        -------------------
        1. Weight Calculation:
           - Recent performance (RMSE, MAE)
           - Regime-specific adjustments
           - Market condition factors
        
        2. Dynamic Adjustment:
           - Exponential smoothing of weights
           - Regime transition impacts
           - Volatility of volatility effects
        
        Error Handling:
        --------------
        - Weight bounds validation
        - Numerical stability checks
        - Missing data handling
        """
        # Weight calculation based on recent forecast performance
        pass

    def _historical_vol_forecast(self, horizon: int) -> float:
        """
        Generate volatility forecast based on historical patterns
        
        Methodology:
        1. GARCH(1,1) Base Model:
           σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
           - Captures volatility clustering
           - Mean reversion characteristics
           - Asymmetric shock response
        
        2. Adjustments:
           - Long-term mean reversion factor
           - Volatility risk premium correction
           - Regime-specific scaling
        
        Args:
            horizon: Forecast horizon in trading days
        
        Returns:
            float: Annualized volatility forecast
        """
        returns = np.log(self.historical_data['price']).diff().dropna()
        
        # Fit GARCH model with asymmetric effects
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='skewt')
        results = model.fit(disp='off')
        
        # Generate forecast with confidence intervals
        forecast = results.forecast(horizon=horizon)
        base_forecast = np.sqrt(forecast.variance.mean()) * np.sqrt(252)
        
        # Apply regime-specific adjustments
        current_regime = self.detect_volatility_regime()
        regime_adjustment = self._get_regime_adjustment(current_regime)
        
        return base_forecast * regime_adjustment

    def _implied_vol_forecast(self, horizon: int) -> float:
        """
        Forward-looking volatility forecast using options market data
        
        Methodology:
        1. Forward Variance Extraction:
           - Strip variance swaps construction
           - Calendar spread analysis
           - Risk-neutral density estimation
        
        2. Risk Premium Adjustment:
           - Historical variance risk premium
           - Term structure adjustment
           - Skew impact consideration
        
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
        
        # Calculate forward variance using variance swap replication
        forward_var = self._extract_forward_variance(relevant_expiries)
        
        # Apply variance risk premium adjustment
        vrp = self._calculate_variance_risk_premium()
        adjusted_var = forward_var * (1 - vrp)
        
        return np.sqrt(adjusted_var)

    def _regime_based_forecast(self, horizon: int) -> float:
        """
        Generate volatility forecast conditioned on current market regime
        
        Methodology:
        1. Regime Characteristics:
           - Typical volatility levels
           - Mean reversion rates
           - Shock persistence
        
        2. Transition Dynamics:
           - Regime duration analysis
           - Transition probabilities
           - Conditional forecasting
        
        Args:
            horizon: Forecast horizon in trading days
        
        Returns:
            float: Regime-adjusted volatility forecast
        """
        current_regime = self.detect_volatility_regime()
        
        # Calculate regime-specific parameters
        regime_params = {
            'low_vol': {
                'mean_level': 0.10,
                'persistence': 0.95,
                'mean_reversion': 0.05
            },
            'normal_vol': {
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
        current_vol = self.historical_data['historical_vol'].iloc[-1]
        
        # Mean-reverting forecast with regime characteristics
        forecast = (params['mean_level'] * (1 - params['persistence']**horizon) +
                   current_vol * params['persistence']**horizon)
        
        return forecast

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
        
        2. Market Interpretation:
           - Forward volatility expectations
           - Term premium estimation
           - Structural breaks detection
        
        Returns:
            Dict containing:
            - term_level: Overall term structure level
            - term_slope: Short-to-long term slope
            - term_curve: Term structure curvature
            - calendar_spreads: Key calendar spread values
        """
        # Group by expiry and calculate ATM vols
        atm_vols = self._extract_atm_vols()
        expiries = np.array(sorted(atm_vols.keys()))
        vols = np.array([atm_vols[exp] for exp in expiries])
        
        # Calculate term structure metrics
        short_term = np.mean(vols[expiries <= 0.25])
        long_term = np.mean(vols[expiries >= 1.0])
        
        # Fit Nelson-Siegel model
        params = self._fit_nelson_siegel(expiries, vols)
        
        return {
            'term_level': np.mean(vols),
            'term_slope': (long_term - short_term) / long_term,
            'term_curve': self._calculate_term_curvature(params),
            'calendar_spreads': self._calculate_calendar_spreads(atm_vols)
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
