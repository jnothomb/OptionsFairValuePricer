import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
import datetime
from .volatility import VolatilityAnalyzer
from scipy.stats import norm  # For VaR calculations

@dataclass
class Portfolio:
    """Container for portfolio data"""
    positions: List[Any]

@dataclass
class MarketData:
    """Container for market data"""
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    volatility_surface: Any
    timestamp: Optional[datetime.datetime] = None

@dataclass
class RiskMetrics:
    """Container for option position risk metrics"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    volga: float  # Vega convexity
    vanna: float  # Delta-vol correlation
    charm: float  # Delta decay
    veta: float   # Vega decay

@dataclass
class RiskThresholds:
    """Risk threshold configuration"""
    max_position_delta: float
    max_position_gamma: float
    max_position_vega: float
    max_position_theta: float
    max_position_size: float
    max_loss_threshold: float

class RiskManager:
    """
    Risk management system for options portfolios
    
    Features:
    1. Greeks Calculation and Aggregation
    2. Scenario Analysis
    3. VaR and Expected Shortfall
    4. Risk Limits Monitoring
    5. Hedging Recommendations
    """
    
    def __init__(self, 
                 positions: pd.DataFrame,
                 vol_analyzer: VolatilityAnalyzer,
                 risk_free_rate: float = 0.02):
        """
        Initialize risk manager
        
        Args:
            positions: DataFrame with option positions
                Required columns:
                - strike: Strike prices
                - expiry: Time to expiration
                - position: Number of contracts (+/- for long/short)
                - option_type: 'call' or 'put'
            vol_analyzer: Volatility analysis engine
            risk_free_rate: Risk-free interest rate
        """
        # Validate required columns
        required_columns = ['strike', 'expiry', 'position', 'option_type']
        if not all(col in positions.columns for col in required_columns):
            raise ValueError(f"Positions data missing required columns: {required_columns}")
        
        # Validate data types
        if not all(positions[['strike', 'expiry', 'position']].dtypes == float):
            raise ValueError("Numeric columns must be float type")
        
        if not all(positions['option_type'].str.lower().isin(['call', 'put'])):
            raise ValueError("option_type must be 'call' or 'put'")
        
        self.positions = positions
        self.vol_analyzer = vol_analyzer
        self.risk_free_rate = risk_free_rate
        self.spot_price = vol_analyzer.spot_price
        
    def calculate_portfolio_greeks(self) -> RiskMetrics:
        """Calculate aggregated Greeks for entire portfolio"""
        total_greeks = {
            'delta': 0.0, 'gamma': 0.0, 'vega': 0.0,
            'theta': 0.0, 'rho': 0.0, 'volga': 0.0,
            'vanna': 0.0, 'charm': 0.0, 'veta': 0.0
        }
        
        for _, pos in self.positions.iterrows():
            greeks = self._calculate_position_greeks(pos)
            size = pos['position']
            
            for greek in total_greeks:
                total_greeks[greek] += size * getattr(greeks, greek)
        
        return RiskMetrics(**total_greeks)
    
    def calculate_var(self, confidence: float = 0.99, 
                     horizon: int = 1) -> float:
        """
        Calculate Value at Risk using Delta-Gamma approximation
        
        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR)
            horizon: Time horizon in days
        
        Returns:
            float: Portfolio VaR estimate
        """
        greeks = self.calculate_portfolio_greeks()
        
        # Get volatility forecast
        vol_forecast = self.vol_analyzer.forecast_volatility(
            horizon=horizon
        )['point_forecast']
        
        # Calculate return distribution parameters
        sigma = vol_forecast * np.sqrt(horizon/252)
        mu = (self.risk_free_rate - 0.5*sigma**2) * horizon/252
        
        # Delta-Gamma VaR
        z_score = norm.ppf(confidence)
        delta_var = greeks.delta * self.spot_price * sigma * z_score
        gamma_var = 0.5 * greeks.gamma * (self.spot_price * sigma)**2
        
        return -(delta_var + gamma_var)
    
    def generate_hedge_recommendations(self) -> Dict[str, float]:
        """
        Generate hedging recommendations based on risk exposures
        
        Returns:
            Dict containing:
            - delta_hedge: Required spot position
            - vega_hedge: Required volatility hedge
            - gamma_hedge: Required convexity hedge
        """
        greeks = self.calculate_portfolio_greeks()
        
        return {
            'delta_hedge': -greeks.delta,
            'vega_hedge': -greeks.vega,
            'gamma_hedge': -greeks.gamma
        }
    
    def run_scenario_analysis(self, 
                            spot_changes: List[float],
                            vol_changes: List[float]) -> pd.DataFrame:
        """
        Run scenario analysis for portfolio value changes
        
        Args:
            spot_changes: List of spot price changes (%)
            vol_changes: List of volatility changes (%)
        
        Returns:
            DataFrame with scenario results
        """
        results = []
        greeks = self.calculate_portfolio_greeks()
        
        for ds in spot_changes:
            for dv in vol_changes:
                # First-order approximation
                delta_pnl = greeks.delta * self.spot_price * ds
                vega_pnl = greeks.vega * dv
                gamma_pnl = 0.5 * greeks.gamma * (self.spot_price * ds)**2
                
                results.append({
                    'spot_change': ds,
                    'vol_change': dv,
                    'pnl_estimate': delta_pnl + vega_pnl + gamma_pnl
                })
        
        return pd.DataFrame(results)
    
    def _calculate_position_greeks(self, position: pd.Series) -> RiskMetrics:
        """
        Calculate Greeks for individual position using Black-Scholes model
        
        Gamma is naturally positive for both calls and puts because:
        Gamma = N'(d1)/(S*σ*√T)
        where N'(d1) is the standard normal PDF (always positive)
        """
        K = position['strike']
        T = position['expiry']
        sigma = self.vol_analyzer._interpolate_vol(
            np.log(K/self.spot_price), T
        )
        
        # Black-Scholes Greeks calculation
        d1 = (np.log(self.spot_price/K) + 
              (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Standard Greeks
        is_call = position['option_type'].lower() == 'call'
        
        # Delta: different for calls and puts
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma: same for calls and puts, naturally positive
        gamma = norm.pdf(d1) / (self.spot_price * sigma * np.sqrt(T))
        
        # Vega: same for calls and puts
        vega = self.spot_price * np.sqrt(T) * norm.pdf(d1)
        
        # Theta: different for calls and puts
        base_theta = (-self.spot_price * norm.pdf(d1) * sigma / 
                     (2*np.sqrt(T)))
        if is_call:
            theta = base_theta - self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
        else:
            theta = base_theta + self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2)
        
        # Rho: different for calls and puts
        if is_call:
            rho = K * T * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2)
        
        # Higher-order Greeks
        volga = vega * d1 * d2 / sigma
        vanna = -norm.pdf(d1) * d2 / sigma
        charm = -norm.pdf(d1) * (2*(self.risk_free_rate-sigma**2)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        veta = -self.spot_price * norm.pdf(d1) * np.sqrt(T) * (
            self.risk_free_rate*d1/(sigma*np.sqrt(T)) - 
            (1 + d1*d2)/(2*T)
        )
        
        # Apply position size (this can make gamma negative for short positions)
        size = position['position']
        return RiskMetrics(
            delta=delta * size,
            gamma=gamma * size,
            vega=vega * size,
            theta=theta * size,
            rho=rho * size,
            volga=volga * size,
            vanna=vanna * size,
            charm=charm * size,
            veta=veta * size
        )
    
    def calculate_expected_shortfall(self, confidence: float = 0.99, horizon: int = 1) -> float:
        """
        Calculate Expected Shortfall (CVaR) using Delta-Gamma approximation
        
        Args:
            confidence: Confidence level (e.g., 0.99 for 99% ES)
            horizon: Time horizon in days
        
        Returns:
            float: Portfolio Expected Shortfall estimate
        """
        greeks = self.calculate_portfolio_greeks()
        
        # Get volatility forecast
        vol_forecast = self.vol_analyzer.forecast_volatility(
            horizon=horizon
        )['point_forecast']
        
        # Calculate return distribution parameters
        sigma = vol_forecast * np.sqrt(horizon/252)
        mu = (self.risk_free_rate - 0.5*sigma**2) * horizon/252
        
        # ES calculation
        z_score = norm.ppf(confidence)
        es_multiplier = norm.pdf(z_score) / (1 - confidence)
        
        delta_es = greeks.delta * self.spot_price * sigma * es_multiplier
        gamma_es = 0.5 * greeks.gamma * (self.spot_price * sigma)**2 * (1 + z_score**2) * es_multiplier
        
        return -(delta_es + gamma_es)
    
    def check_risk_limits(self, limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if portfolio risks exceed specified limits
        
        Args:
            limits: Dictionary of risk limits, e.g.,
                {'delta': 100, 'vega': 1000, 'var_99': 1000000}
        
        Returns:
            Dict: Boolean indicators for each limit breach
        """
        greeks = self.calculate_portfolio_greeks()
        var_99 = self.calculate_var(confidence=0.99)
        
        breaches = {}
        for metric, limit in limits.items():
            if metric == 'var_99':
                breaches[metric] = abs(var_99) > limit
            elif hasattr(greeks, metric):
                breaches[metric] = abs(getattr(greeks, metric)) > limit
                
        return breaches
    
    def calculate_stress_test(self, scenarios: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Perform stress testing under different market scenarios
        
        Args:
            scenarios: List of scenario dictionaries, each containing:
                - spot_shock: Percentage change in spot price
                - vol_shock: Percentage change in volatility
                - rate_shock: Absolute change in rates (bps)
        
        Returns:
            DataFrame with scenario results
        """
        results = []
        base_greeks = self.calculate_portfolio_greeks()
        
        for scenario in scenarios:
            spot_shock = scenario.get('spot_shock', 0)
            vol_shock = scenario.get('vol_shock', 0)
            rate_shock = scenario.get('rate_shock', 0) / 10000  # bps to decimal
            
            # Calculate P&L components
            delta_pnl = base_greeks.delta * self.spot_price * spot_shock
            gamma_pnl = 0.5 * base_greeks.gamma * (self.spot_price * spot_shock)**2
            vega_pnl = base_greeks.vega * vol_shock
            theta_pnl = base_greeks.theta * (1/252)  # Daily theta
            rho_pnl = base_greeks.rho * rate_shock
            
            total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
            
            results.append({
                'scenario': str(scenario),
                'total_pnl': total_pnl,
                'delta_pnl': delta_pnl,
                'gamma_pnl': gamma_pnl,
                'vega_pnl': vega_pnl,
                'theta_pnl': theta_pnl,
                'rho_pnl': rho_pnl
            })
        
        return pd.DataFrame(results)
    
    def calculate_portfolio_sensitivity(self, greek: str, 
                                     shock_range: List[float]) -> pd.DataFrame:
        """
        Calculate portfolio sensitivity to specific risk factor
        
        Args:
            greek: Risk factor to shock ('delta', 'vega', etc.)
            shock_range: List of shock values to apply
        
        Returns:
            DataFrame with sensitivity analysis results
        """
        base_greeks = self.calculate_portfolio_greeks()
        results = []
        
        for shock in shock_range:
            if greek == 'delta':
                pnl = base_greeks.delta * self.spot_price * shock
            elif greek == 'gamma':
                pnl = 0.5 * base_greeks.gamma * (self.spot_price * shock)**2
            elif greek == 'vega':
                pnl = base_greeks.vega * shock
            elif greek == 'theta':
                pnl = base_greeks.theta * shock
            elif greek == 'rho':
                pnl = base_greeks.rho * shock
            else:
                raise ValueError(f"Unsupported greek: {greek}")
            
            results.append({
                'shock': shock,
                'pnl': pnl
            })
        
        return pd.DataFrame(results)
    
    def optimize_hedge_ratios(self, constraints: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize hedge ratios subject to risk constraints
        
        Args:
            constraints: Dictionary of maximum allowed exposures
                e.g., {'delta': 0.1, 'gamma': 1.0, 'vega': 50}
        
        Returns:
            Dict of optimal hedge ratios for different instruments
        """
        from scipy.optimize import minimize
        
        def objective(x):
            # Minimize hedging cost
            return np.sum(np.abs(x))
        
        def constraint_function(x):
            # Calculate residual exposures after hedging
            residual_exposures = []
            for greek, limit in constraints.items():
                if hasattr(base_greeks, greek):
                    exposure = getattr(base_greeks, greek) + x[0]  # Simplified
                    residual_exposures.append(abs(exposure) - limit)
            return np.array(residual_exposures)
        
        base_greeks = self.calculate_portfolio_greeks()
        
        # Optimize (simplified version)
        result = minimize(
            objective,
            x0=np.zeros(len(constraints)),
            constraints={'type': 'ineq', 'fun': lambda x: -constraint_function(x)}
        )
        
        return dict(zip(constraints.keys(), result.x))
    
    def calculate_margin_requirements(self, margin_params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate initial and maintenance margin requirements
        
        Args:
            margin_params: Dictionary of margin parameters
                e.g., {'im_multiplier': 1.5, 'mm_ratio': 0.75}
        
        Returns:
            Dict containing margin requirements
        """
        greeks = self.calculate_portfolio_greeks()
        var_99 = self.calculate_var(confidence=0.99)
        
        # Calculate initial margin components
        exposure_margin = abs(greeks.delta * self.spot_price)
        volatility_margin = abs(greeks.vega * 0.01)  # 1 vol point
        var_margin = abs(var_99)
        
        # Total initial margin
        im_multiplier = margin_params.get('im_multiplier', 1.5)
        initial_margin = im_multiplier * (exposure_margin + volatility_margin + var_margin)
        
        # Maintenance margin
        mm_ratio = margin_params.get('mm_ratio', 0.75)
        maintenance_margin = initial_margin * mm_ratio
        
        return {
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
            'exposure_component': exposure_margin,
            'volatility_component': volatility_margin,
            'var_component': var_margin
        }
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        
        Returns:
            Dict containing:
            - Portfolio Greeks
            - VaR and ES metrics
            - Risk limit breaches
            - Margin requirements
            - Hedge recommendations
        """
        greeks = self.calculate_portfolio_greeks()
        var_99 = self.calculate_var(confidence=0.99)
        es_99 = self.calculate_expected_shortfall(confidence=0.99)
        
        # Standard risk limits
        standard_limits = {
            'delta': 100,
            'gamma': 1.0,
            'vega': 1000,
            'var_99': 1000000
        }
        
        limit_breaches = self.check_risk_limits(standard_limits)
        hedge_recs = self.generate_hedge_recommendations()
        margin_reqs = self.calculate_margin_requirements({'im_multiplier': 1.5})
        
        return {
            'timestamp': pd.Timestamp.now(),
            'portfolio_greeks': {
                greek: getattr(greeks, greek)
                for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']
            },
            'risk_metrics': {
                'var_99': var_99,
                'es_99': es_99,
                'limit_breaches': limit_breaches
            },
            'margin_requirements': margin_reqs,
            'hedge_recommendations': hedge_recs
        }
    
    def _validate_portfolio(self, portfolio: Portfolio) -> None:
        """
        Validate portfolio structure while preserving market opportunities
        
        Data Integrity Checks:
        1. Portfolio structure and completeness
        2. Data types and formats
        3. Basic consistency
        
        Market Opportunity Detection:
        1. Unusual position sizes
        2. Concentrated exposures
        3. Arbitrage opportunities
        """
        if not portfolio.positions:
            raise ValueError("Portfolio cannot be empty")
            
        # Data Integrity Checks
        for position in portfolio.positions:
            if not hasattr(position, 'quantity'):
                raise ValueError("Position missing quantity attribute")
            if not hasattr(position, 'instrument'):
                raise ValueError("Position missing instrument attribute")
                
            # Validate instrument attributes
            instrument = position.instrument
            required_attrs = ['strike', 'expiry', 'option_type']
            missing_attrs = [attr for attr in required_attrs 
                           if not hasattr(instrument, attr)]
            if missing_attrs:
                raise ValueError(
                    f"Instrument missing required attributes: {missing_attrs}"
                )
            
            # Type validation
            if not isinstance(position.quantity, (int, float)):
                raise ValueError("Position quantity must be numeric")
                
            if not isinstance(instrument.strike, (int, float)):
                raise ValueError("Strike price must be numeric")
                
            if not isinstance(instrument.expiry, (int, float)):
                raise ValueError("Expiry must be numeric")
                
            if instrument.option_type.lower() not in ['call', 'put']:
                raise ValueError(
                    f"Invalid option type: {instrument.option_type}. "
                    "Must be 'call' or 'put'"
                )
                
        # Market Opportunity Detection
        total_value = sum(abs(p.quantity * p.instrument.strike) 
                         for p in portfolio.positions)
        position_values = [
            (p, abs(p.quantity * p.instrument.strike)) 
            for p in portfolio.positions
        ]
        
        for position, value in position_values:
            if value > total_value * 0.2:  # 20% concentration
                warnings.warn(
                    f"Large position detected: {value/total_value:.1%} of portfolio\n"
                    f"Strike: {position.instrument.strike}, "
                    f"Type: {position.instrument.option_type}"
                )
                
        # Check for potential arbitrage opportunities
        calls = [p for p in portfolio.positions 
                if p.instrument.option_type.lower() == 'call']
        puts = [p for p in portfolio.positions 
               if p.instrument.option_type.lower() == 'put']
        
        # Check put-call parity opportunities
        for call in calls:
            matching_puts = [
                p for p in puts 
                if abs(p.instrument.strike - call.instrument.strike) < 0.01 and
                abs(p.instrument.expiry - call.instrument.expiry) < 0.01
            ]
            
            if matching_puts:
                warnings.warn(
                    f"Potential put-call parity opportunity:\n"
                    f"Call: Strike={call.instrument.strike}, "
                    f"Expiry={call.instrument.expiry}\n"
                    f"Put: Strike={matching_puts[0].instrument.strike}, "
                    f"Expiry={matching_puts[0].instrument.expiry}"
                )
                
    def _validate_risk_limits(self, risk_limits: Dict[str, float]) -> None:
        """
        Validate risk limits while preserving trading flexibility
        
        Data Integrity Checks:
        1. Required limits presence
        2. Data types
        3. Basic consistency
        
        Risk Management Insights:
        1. Unusual limit levels
        2. Risk concentration
        3. Hedging opportunities
        """
        required_limits = [
            'max_position_size',
            'max_portfolio_delta',
            'max_portfolio_gamma',
            'max_portfolio_vega',
            'max_portfolio_theta',
            'max_portfolio_rho'
        ]
        
        # Check required limits
        missing_limits = [limit for limit in required_limits 
                         if limit not in risk_limits]
        if missing_limits:
            raise ValueError(f"Missing required risk limits: {missing_limits}")
            
        # Type validation
        for limit_name, limit_value in risk_limits.items():
            if not isinstance(limit_value, (int, float)):
                raise ValueError(f"Risk limit {limit_name} must be numeric")
                
        # Risk Management Insights
        typical_limits = {
            'max_position_size': 0.2,  # 20% of portfolio
            'max_portfolio_delta': 0.3,  # 30% of portfolio
            'max_portfolio_gamma': 0.1,  # 10% of portfolio
            'max_portfolio_vega': 0.2,   # 20% of portfolio
            'max_portfolio_theta': 0.05,  # 5% of portfolio
            'max_portfolio_rho': 0.1      # 10% of portfolio
        }
        
        for limit_name, typical_value in typical_limits.items():
            if limit_name in risk_limits:
                actual_value = abs(risk_limits[limit_name])
                if actual_value > 2 * typical_value:
                    warnings.warn(
                        f"Unusually high {limit_name}: {actual_value:.1%} "
                        f"(typical: {typical_value:.1%})"
                    )
                    
        # Check for asymmetric limits
        if ('max_long_delta' in risk_limits and 
            'max_short_delta' in risk_limits):
            long_delta = abs(risk_limits['max_long_delta'])
            short_delta = abs(risk_limits['max_short_delta'])
            
            if max(long_delta, short_delta) > 2 * min(long_delta, short_delta):
                warnings.warn(
                    f"Highly asymmetric delta limits detected: "
                    f"Long={long_delta:.1%}, Short={short_delta:.1%}"
                )
                
    def _validate_market_data(self, market_data: MarketData) -> None:
        """
        Validate market data while preserving anomaly detection
        
        Data Integrity Checks:
        1. Required data presence
        2. Data types and formats
        3. Basic consistency
        
        Market Signal Detection:
        1. Price anomalies
        2. Volatility patterns
        3. Trading opportunities
        """
        required_data = [
            'spot_price',
            'risk_free_rate',
            'dividend_yield',
            'volatility_surface'
        ]
        
        # Check required data
        missing_data = [data for data in required_data 
                       if not hasattr(market_data, data)]
        if missing_data:
            raise ValueError(f"Missing required market data: {missing_data}")
            
        # Type validation
        if not isinstance(market_data.spot_price, (int, float)):
            raise ValueError("Spot price must be numeric")
            
        if not isinstance(market_data.risk_free_rate, (int, float)):
            raise ValueError("Risk-free rate must be numeric")
            
        if not isinstance(market_data.dividend_yield, (int, float)):
            raise ValueError("Dividend yield must be numeric")
            
        # Validate volatility surface interface
        if not hasattr(market_data.volatility_surface, 'get_implied_vol'):
            raise ValueError(
                "Volatility surface must implement get_implied_vol method"
            )
            
        # Market Signal Detection
        typical_conditions = {
            'risk_free_rate': 0.03,  # 3%
            'dividend_yield': 0.02   # 2%
        }
        
        if abs(market_data.risk_free_rate) > 2 * typical_conditions['risk_free_rate']:
            warnings.warn(
                f"Unusual risk-free rate: {market_data.risk_free_rate:.1%} "
                f"(typical: {typical_conditions['risk_free_rate']:.1%})"
            )
            
        if market_data.dividend_yield > 2 * typical_conditions['dividend_yield']:
            warnings.warn(
                f"High dividend yield: {market_data.dividend_yield:.1%} "
                f"(typical: {typical_conditions['dividend_yield']:.1%})"
            )
            
        # Check data freshness if timestamp available
        if market_data.timestamp:
            max_age = datetime.timedelta(minutes=15)
            if datetime.datetime.now() - market_data.timestamp > max_age:
                warnings.warn(
                    f"Market data is more than {max_age} old. "
                    "Consider refreshing for time-sensitive trades."
                )