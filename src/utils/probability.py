import numpy as np
from scipy.stats import norm
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable
import warnings

class DistributionType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    MIXTURE = "mixture"

class ProbabilityCalculator:
    """
    Advanced probability calculator for option pricing and risk analysis.
    
    This class implements various probability calculations needed for options trading:
    1. Price movement probabilities using different distributions
    2. Expected values and moments
    3. Confidence intervals and tail probabilities
    4. Risk-neutral probabilities for pricing
    5. Mixture models for regime-switching scenarios
    """
    
    def __init__(self, spot_price: float, volatility: float, time_horizon: float, risk_free_rate: float):
        """
        Initialize the probability calculator.
        
        Args:
            spot_price: Current price of the underlying asset
            volatility: Annualized volatility
            time_horizon: Time horizon in years
            risk_free_rate: Annual risk-free rate
        
        Raises:
            ValueError: If any input parameters are invalid
        """
        # Input validation
        if spot_price <= 0:
            raise ValueError("Spot price must be positive")
        if volatility < 0:  # Allow zero volatility for boundary conditions
            raise ValueError("Volatility must be non-negative")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("Risk-free rate must be a number")
            
        self.spot_price = spot_price
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.risk_free_rate = risk_free_rate
        
        # Initialize price grid for numerical calculations
        self._initialize_price_grid()
        
    def _initialize_price_grid(self):
        """Initialize price grid based on current parameters."""
        # For zero volatility, use tight grid around forward price
        if self.volatility == 0:
            forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
            self.price_range = np.linspace(forward * 0.999, forward * 1.001, 1000)
        else:
            # Set grid boundaries based on confidence interval
            std_dev = self.volatility * np.sqrt(self.time_horizon)
            z_score = norm.ppf(0.9999)  # 99.99% confidence
            
            lower_bound = self.spot_price * np.exp(-z_score * std_dev)
            upper_bound = self.spot_price * np.exp(z_score * std_dev)
            
            self.price_range = np.linspace(lower_bound, upper_bound, 1000)
            
        self.price_step = (self.price_range[-1] - self.price_range[0]) / 999
        
    def calculate_probability_density(
        self,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        mixture_weights: Optional[List[float]] = None,
        mixture_params: Optional[List[Tuple[float, float]]] = None,
        time_horizon: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate probability density for future price distribution.
        
        Args:
            distribution_type: Type of distribution to use
            mixture_weights: Weights for mixture components (if using mixture)
            mixture_params: Parameters for mixture components [(vol1, drift1), ...]
            time_horizon: Optional override for time horizon
            
        Returns:
            np.ndarray: Probability density values
        """
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            self._initialize_price_grid()
        
        try:
            if distribution_type == DistributionType.NORMAL:
                pdf = self._normal_density()
            elif distribution_type == DistributionType.LOGNORMAL:
                pdf = self._lognormal_density()
            elif distribution_type == DistributionType.MIXTURE:
                if not mixture_weights or not mixture_params:
                    raise ValueError("Mixture weights and parameters required for mixture distribution")
                if abs(sum(mixture_weights) - 1.0) > 1e-10:
                    raise ValueError("Mixture weights must sum to 1")
                pdf = self._mixture_density(mixture_weights, mixture_params)
            else:
                raise ValueError(f"Unsupported distribution type: {distribution_type}")
                
            # Handle zero volatility case
            if self.volatility == 0:
                forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                idx = np.argmin(np.abs(self.price_range - forward))
                pdf = np.zeros_like(self.price_range)
                pdf[idx] = 1.0 / self.price_step
                return pdf
                
            # Ensure proper normalization
            total_prob = np.sum(pdf) * self.price_step
            if total_prob > 0:
                pdf = pdf / total_prob
            return pdf
            
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                self._initialize_price_grid()
        
    def _normal_density(self) -> np.ndarray:
        """Calculate normal distribution density."""
        if self.volatility == 0:
            return self._zero_vol_density()
            
        mu = self.spot_price + (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_horizon
        sigma = self.volatility * np.sqrt(self.time_horizon)
        return norm.pdf(self.price_range, mu, sigma)
        
    def _lognormal_density(self) -> np.ndarray:
        """Calculate lognormal distribution density (risk-neutral)."""
        if self.volatility == 0:
            return self._zero_vol_density()
            
        log_prices = np.log(self.price_range / self.spot_price)
        mu = (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_horizon
        sigma = self.volatility * np.sqrt(self.time_horizon)
        return norm.pdf(log_prices, mu, sigma) / self.price_range
        
    def _zero_vol_density(self) -> np.ndarray:
        """Calculate density for zero volatility case."""
        forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
        pdf = np.zeros_like(self.price_range)
        idx = np.argmin(np.abs(self.price_range - forward))
        pdf[idx] = 1.0 / self.price_step
        return pdf
        
    def _mixture_density(
        self,
        weights: List[float],
        params: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Calculate mixture distribution density."""
        if self.volatility == 0:
            return self._zero_vol_density()
            
        density = np.zeros_like(self.price_range)
        
        # Store original parameters
        orig_vol = self.volatility
        orig_rate = self.risk_free_rate
        
        try:
            # Calculate each component on the same grid
            for w, (vol, drift) in zip(weights, params):
                self.volatility = vol
                self.risk_free_rate = drift
                # Use lognormal density without reinitializing grid
                component = self._lognormal_density()
                # Ensure proper normalization of component
                comp_sum = np.sum(component) * self.price_step
                if comp_sum > 0:
                    component = component / comp_sum
                density += w * component
                
            return density
            
        finally:
            # Restore original parameters
            self.volatility = orig_vol
            self.risk_free_rate = orig_rate
        
    def calculate_probability_range(
        self,
        lower_price: float,
        upper_price: float,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> float:
        """
        Calculate probability of price ending between two levels.
        
        Args:
            lower_price: Lower price boundary
            upper_price: Upper price boundary
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            float: Probability of price ending in the range
        """
        if lower_price >= upper_price:
            raise ValueError("Upper price must be greater than lower price")
            
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            if self.volatility == 0:
                forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                return float(lower_price <= forward <= upper_price)
                
            if distribution_type == DistributionType.LOGNORMAL:
                # Use log-space calculations for better numerical stability
                log_lower = np.log(lower_price / self.spot_price)
                log_upper = np.log(upper_price / self.spot_price)
                mu = (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_horizon
                sigma = self.volatility * np.sqrt(self.time_horizon)
                return norm.cdf(log_upper, mu, sigma) - norm.cdf(log_lower, mu, sigma)
            else:
                # Numerical integration for other distributions
                pdf = self.calculate_probability_density(distribution_type)
                mask = (self.price_range >= lower_price) & (self.price_range <= upper_price)
                return np.sum(pdf[mask]) * self.price_step
                
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                
    def calculate_expected_price(
        self,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> float:
        """
        Calculate expected future price.
        
        For lognormal distribution, this is S0 * exp(rT) in the risk-neutral measure.
        
        Args:
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            float: Expected price
        """
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            # For all distributions in risk-neutral measure
            return self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                
    def calculate_price_variance(
        self,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> float:
        """
        Calculate variance of future price distribution.
        
        Args:
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            float: Price variance
        """
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            if self.volatility == 0:
                return 0.0
                
            if distribution_type == DistributionType.LOGNORMAL:
                # Analytical solution for lognormal
                return (self.spot_price * np.exp(self.risk_free_rate * self.time_horizon))**2 * \
                       (np.exp(self.volatility**2 * self.time_horizon) - 1)
            else:
                # Numerical calculation for other distributions
                pdf = self.calculate_probability_density(distribution_type)
                expected_price = self.calculate_expected_price(distribution_type)
                return np.sum(((self.price_range - expected_price)**2) * pdf) * self.price_step
                
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                
    def calculate_confidence_interval(
        self,
        confidence: float,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for future price.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            Tuple[float, float]: Lower and upper bounds of the interval
        """
        if not 0 < confidence < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            if self.volatility == 0:
                forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                return forward, forward
                
            alpha = (1 - confidence) / 2
            
            if distribution_type == DistributionType.LOGNORMAL:
                # Analytical solution for lognormal
                mu = (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_horizon
                sigma = self.volatility * np.sqrt(self.time_horizon)
                z_score = norm.ppf(1 - alpha)
                
                # Ensure symmetry in log space
                log_center = mu
                log_width = z_score * sigma
                
                lower = self.spot_price * np.exp(log_center - log_width)
                upper = self.spot_price * np.exp(log_center + log_width)
                return lower, upper
            else:
                # Numerical solution for other distributions
                pdf = self.calculate_probability_density(distribution_type)
                cdf = np.cumsum(pdf) * self.price_step
                cdf = cdf / cdf[-1]  # Ensure proper normalization
                
                lower_idx = np.searchsorted(cdf, alpha)
                upper_idx = np.searchsorted(cdf, 1 - alpha)
                
                return self.price_range[lower_idx], self.price_range[upper_idx]
                
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                
    def calculate_tail_probability(
        self,
        threshold: float,
        tail: str = 'lower',
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> float:
        """
        Calculate tail probability (probability of price below/above threshold).
        
        Args:
            threshold: Price threshold
            tail: 'lower' or 'upper'
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            float: Tail probability
        """
        if tail not in ['lower', 'upper']:
            raise ValueError("Tail must be 'lower' or 'upper'")
            
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            if self.volatility == 0:
                forward = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                if tail == 'lower':
                    return float(forward < threshold)
                else:
                    return float(forward > threshold)
                    
            if distribution_type == DistributionType.LOGNORMAL:
                # Analytical solution for lognormal
                log_threshold = np.log(threshold / self.spot_price)
                mu = (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_horizon
                sigma = self.volatility * np.sqrt(self.time_horizon)
                
                if tail == 'lower':
                    return norm.cdf(log_threshold, mu, sigma)
                else:
                    return 1 - norm.cdf(log_threshold, mu, sigma)
            else:
                # Numerical solution for other distributions
                pdf = self.calculate_probability_density(distribution_type)
                if tail == 'lower':
                    mask = self.price_range <= threshold
                else:
                    mask = self.price_range >= threshold
                return np.sum(pdf[mask]) * self.price_step
                
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
                
    def calculate_distribution_moments(
        self,
        distribution_type: DistributionType = DistributionType.LOGNORMAL,
        time_horizon: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate distribution moments (mean, variance, skewness, kurtosis).
        
        Args:
            distribution_type: Type of distribution to use
            time_horizon: Optional override for time horizon
            
        Returns:
            Dict[str, float]: Distribution moments
        """
        if time_horizon is not None:
            old_time = self.time_horizon
            self.time_horizon = time_horizon
            
        try:
            if self.volatility == 0:
                mean = self.spot_price * np.exp(self.risk_free_rate * self.time_horizon)
                return {
                    'mean': mean,
                    'variance': 0.0,
                    'skewness': 0.0,
                    'kurtosis': float('inf')  # Point mass has infinite kurtosis
                }
                
            if distribution_type == DistributionType.LOGNORMAL:
                # Analytical solutions for lognormal
                T = self.time_horizon
                r = self.risk_free_rate
                sigma = self.volatility
                S0 = self.spot_price
                
                mean = S0 * np.exp(r * T)
                variance = S0**2 * np.exp(2*r*T) * (np.exp(sigma**2 * T) - 1)
                skewness = (np.exp(sigma**2 * T) + 2) * np.sqrt(np.exp(sigma**2 * T) - 1)
                kurtosis = np.exp(4*sigma**2 * T) + 2*np.exp(3*sigma**2 * T) + \
                          3*np.exp(2*sigma**2 * T) - 3
            else:
                # Numerical calculations for other distributions
                pdf = self.calculate_probability_density(distribution_type)
                mean = np.sum(self.price_range * pdf) * self.price_step
                
                central_moments = []
                for k in range(2, 5):
                    moment = np.sum(((self.price_range - mean)**k) * pdf) * self.price_step
                    central_moments.append(moment)
                
                variance = central_moments[0]
                skewness = central_moments[1] / variance**1.5 if variance > 0 else 0
                kurtosis = central_moments[2] / variance**2 if variance > 0 else float('inf')
                
            return {
                'mean': mean,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        finally:
            if time_horizon is not None:
                self.time_horizon = old_time
