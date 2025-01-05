import pytest
import numpy as np
from scipy.stats import norm
from src.utils.probability import ProbabilityCalculator, DistributionType

class TestProbabilityCalculator:
    """Test suite for probability calculations with focus on mathematical correctness"""
    
    @pytest.fixture
    def calculator(self):
        """Initialize calculator with standard market parameters"""
        return ProbabilityCalculator(
            spot_price=100.0,
            volatility=0.2,  # 20% annual vol
            time_horizon=1/12,  # 1 month
            risk_free_rate=0.05  # 5% annual rate
        )
    
    def test_risk_neutral_measure(self, calculator):
        """Test risk-neutral measure properties"""
        # Expected value should be S0 * exp(rT)
        expected_price = calculator.calculate_expected_price()
        theoretical_price = calculator.spot_price * np.exp(calculator.risk_free_rate * calculator.time_horizon)
        assert np.isclose(expected_price, theoretical_price, rtol=1e-10)
        
        # Martingale property: discounted price should be a martingale
        discount_factor = np.exp(-calculator.risk_free_rate * calculator.time_horizon)
        discounted_expected = expected_price * discount_factor
        assert np.isclose(discounted_expected, calculator.spot_price, rtol=1e-10)
    
    def test_probability_axioms(self, calculator):
        """Test fundamental probability axioms"""
        # Non-negativity
        pdf = calculator.calculate_probability_density()
        assert np.all(pdf >= 0)
        
        # Unit measure
        total_prob = np.sum(pdf) * calculator.price_step
        assert np.isclose(total_prob, 1.0, rtol=1e-3)
        
        # Complement rule
        threshold = calculator.spot_price
        lower_tail = calculator.calculate_tail_probability(threshold, 'lower')
        upper_tail = calculator.calculate_tail_probability(threshold, 'upper')
        assert np.isclose(lower_tail + upper_tail, 1.0, rtol=1e-10)
    
    def test_lognormal_properties(self, calculator):
        """Test specific properties of lognormal distribution"""
        T = calculator.time_horizon
        r = calculator.risk_free_rate
        sigma = calculator.volatility
        S0 = calculator.spot_price
        
        moments = calculator.calculate_distribution_moments()
        
        # Mean
        assert np.isclose(moments['mean'], S0 * np.exp(r * T), rtol=1e-10)
        
        # Variance
        theoretical_var = (S0 * np.exp(r * T))**2 * (np.exp(sigma**2 * T) - 1)
        assert np.isclose(moments['variance'], theoretical_var, rtol=1e-10)
        
        # Skewness
        theoretical_skew = (np.exp(sigma**2 * T) + 2) * np.sqrt(np.exp(sigma**2 * T) - 1)
        assert np.isclose(moments['skewness'], theoretical_skew, rtol=1e-10)
        
        # Kurtosis
        theoretical_kurt = np.exp(4*sigma**2 * T) + 2*np.exp(3*sigma**2 * T) + \
                         3*np.exp(2*sigma**2 * T) - 3
        assert np.isclose(moments['kurtosis'], theoretical_kurt, rtol=1e-10)
    
    def test_confidence_intervals(self, calculator):
        """Test confidence interval properties"""
        confidence_levels = [0.68, 0.95, 0.99]
        
        for conf in confidence_levels:
            lower, upper = calculator.calculate_confidence_interval(conf)
            
            # Verify interval contains correct probability mass
            prob_in_interval = calculator.calculate_probability_range(lower, upper)
            assert np.isclose(prob_in_interval, conf, rtol=1e-2)
            
            # Verify symmetry in log space around risk-neutral drift
            mu = (calculator.risk_free_rate - 0.5 * calculator.volatility**2) * calculator.time_horizon
            log_lower = np.log(lower / calculator.spot_price) - mu
            log_upper = np.log(upper / calculator.spot_price) - mu
            assert np.isclose(abs(log_lower), abs(log_upper), rtol=1e-10)
    
    def test_mixture_distribution(self, calculator):
        """Test mixture distribution properties"""
        weights = [0.6, 0.4]
        params = [(0.2, 0.05), (0.4, 0.1)]  # [(vol1, drift1), (vol2, drift2)]
        
        mixture_pdf = calculator.calculate_probability_density(
            distribution_type=DistributionType.MIXTURE,
            mixture_weights=weights,
            mixture_params=params
        )
        
        # Test probability axioms for mixture
        assert np.all(mixture_pdf >= 0)
        assert np.isclose(np.sum(mixture_pdf) * calculator.price_step, 1.0, rtol=1e-3)
        
        # Test convexity property by verifying mixture is a weighted sum
        # First calculate individual components with their own parameters
        components = []
        for vol, drift in params:
            # Store original parameters
            orig_vol = calculator.volatility
            orig_rate = calculator.risk_free_rate
            
            try:
                calculator.volatility = vol
                calculator.risk_free_rate = drift
                component = calculator.calculate_probability_density()
                components.append(component)
            finally:
                calculator.volatility = orig_vol
                calculator.risk_free_rate = orig_rate
        
        # Calculate weighted sum of components
        weighted_sum = np.zeros_like(mixture_pdf)
        for w, comp in zip(weights, components):
            weighted_sum += w * comp
            
        # Verify mixture is close to weighted sum
        assert np.allclose(mixture_pdf, weighted_sum, rtol=1e-3)
    
    def test_time_scaling(self, calculator):
        """Test proper scaling with time"""
        # Variance should scale linearly with time
        var1 = calculator.calculate_price_variance(time_horizon=1/252)  # 1 day
        var2 = calculator.calculate_price_variance(time_horizon=2/252)  # 2 days
        assert np.isclose(var2/var1, 2.0, rtol=1e-2)
        
        # Confidence intervals should widen with sqrt(time)
        conf = 0.95
        lower1, upper1 = calculator.calculate_confidence_interval(conf, time_horizon=1/12)
        lower2, upper2 = calculator.calculate_confidence_interval(conf, time_horizon=1/3)
        
        width1 = np.log(upper1/lower1)
        width2 = np.log(upper2/lower2)
        assert np.isclose(width2/width1, np.sqrt(4), rtol=1e-2)  # sqrt(T2/T1)
    
    def test_numerical_stability(self, calculator):
        """Test numerical stability in extreme scenarios"""
        # Test very short time horizons
        calculator.time_horizon = 1/252  # 1 day
        pdf = calculator.calculate_probability_density()
        assert np.all(np.isfinite(pdf))
        assert not np.any(np.isnan(pdf))
        
        # Test very long time horizons
        calculator.time_horizon = 10.0  # 10 years
        pdf = calculator.calculate_probability_density()
        assert np.all(np.isfinite(pdf))
        assert not np.any(np.isnan(pdf))
        
        # Test high volatility
        calculator.volatility = 1.0  # 100% vol
        pdf = calculator.calculate_probability_density()
        assert np.all(np.isfinite(pdf))
        assert not np.any(np.isnan(pdf))
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        with pytest.raises(ValueError, match="must be positive"):
            ProbabilityCalculator(
                spot_price=-100,
                volatility=0.2,
                time_horizon=1/12,
                risk_free_rate=0.05
            )
        
        with pytest.raises(ValueError, match="must be non-negative"):
            ProbabilityCalculator(
                spot_price=100,
                volatility=-0.2,
                time_horizon=1/12,
                risk_free_rate=0.05
            )
            
        with pytest.raises(ValueError, match="must be positive"):
            ProbabilityCalculator(
                spot_price=100,
                volatility=0.2,
                time_horizon=-1/12,
                risk_free_rate=0.05
            )
            
        calculator = ProbabilityCalculator(100, 0.2, 1/12, 0.05)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            calculator.calculate_confidence_interval(1.5)
        
        with pytest.raises(ValueError, match="must sum to 1"):
            calculator.calculate_probability_density(
                distribution_type=DistributionType.MIXTURE,
                mixture_weights=[0.7, 0.7],
                mixture_params=[(0.2, 0.05), (0.4, 0.1)]
            )
    
    def test_boundary_conditions(self, calculator):
        """Test behavior at boundary conditions"""
        # Zero volatility should give deterministic forward price
        calculator.volatility = 0.0  # Use exactly zero
        expected_price = calculator.calculate_expected_price()
        pdf = calculator.calculate_probability_density()
        
        max_prob_price = calculator.price_range[np.argmax(pdf)]
        assert np.isclose(max_prob_price, expected_price, rtol=1e-3)
        
        # Zero time should give spot price
        calculator.time_horizon = 1e-10
        expected_price = calculator.calculate_expected_price()
        assert np.isclose(expected_price, calculator.spot_price, rtol=1e-10)
        
        # Zero rate should preserve martingale property
        calculator.risk_free_rate = 0.0
        expected_price = calculator.calculate_expected_price()
        assert np.isclose(expected_price, calculator.spot_price, rtol=1e-10) 