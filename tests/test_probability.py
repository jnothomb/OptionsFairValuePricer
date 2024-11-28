import pytest
import numpy as np
from src.utils.probability import ProbabilityCalculator, DistributionType

class TestProbability:
    """Test suite for probability calculations"""

    @pytest.fixture
    def calculator(self):
        """Initialize probability calculator"""
        return ProbabilityCalculator(
            spot_price=100,
            volatility=0.2,
            risk_free_rate=0.05,
            time_horizon=30/365  # 30 days
        )

    def test_initialization(self, calculator):
        """Test proper initialization"""
        assert calculator.spot_price == 100
        assert calculator.volatility == 0.2
        assert calculator.risk_free_rate == 0.05
        assert calculator.time_horizon == pytest.approx(30/365)

    @pytest.mark.parametrize("distribution_type", [
        DistributionType.NORMAL,
        DistributionType.LOGNORMAL,
        DistributionType.MIXTURE
    ])
    def test_distribution_types(self, calculator, distribution_type):
        """Test different distribution types"""
        probabilities = calculator.calculate_probability_density(
            distribution_type=distribution_type
        )
        assert len(probabilities) > 0
        assert np.all(probabilities >= 0)
        assert np.isclose(np.sum(probabilities) * calculator.price_step, 1.0, rtol=1e-2)

    def test_probability_ranges(self, calculator):
        """Test probability calculations for different ranges"""
        lower_bound = 90
        upper_bound = 110
        
        prob = calculator.calculate_probability_range(lower_bound, upper_bound)
        assert 0 <= prob <= 1
        
        # Test cumulative probability
        cdf = calculator.calculate_cumulative_probability()
        assert np.all(np.diff(cdf) >= 0)  # Should be monotonically increasing
        assert np.isclose(cdf[-1], 1.0, rtol=1e-3)

    def test_expected_values(self, calculator):
        """Test expected value calculations"""
        expected_price = calculator.calculate_expected_price()
        assert expected_price > 0
        
        variance = calculator.calculate_price_variance()
        assert variance > 0

    def test_confidence_intervals(self, calculator):
        """Test confidence interval calculations"""
        intervals = [0.68, 0.95, 0.99]  # Common confidence levels
        
        for confidence in intervals:
            lower, upper = calculator.calculate_confidence_interval(confidence)
            assert lower < upper
            assert lower > 0
            
            # Verify probability mass within interval
            prob_mass = calculator.calculate_probability_range(lower, upper)
            assert np.isclose(prob_mass, confidence, rtol=1e-2)

    def test_tail_probabilities(self, calculator):
        """Test tail probability calculations"""
        threshold = 90  # Lower tail
        lower_tail_prob = calculator.calculate_tail_probability(threshold, tail='lower')
        assert 0 <= lower_tail_prob <= 1
        
        threshold = 110  # Upper tail
        upper_tail_prob = calculator.calculate_tail_probability(threshold, tail='upper')
        assert 0 <= upper_tail_prob <= 1

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        with pytest.raises(ValueError):
            ProbabilityCalculator(
                spot_price=-100,  # Negative price
                volatility=0.2,
                risk_free_rate=0.05,
                time_horizon=30/365
            )
        
        with pytest.raises(ValueError):
            ProbabilityCalculator(
                spot_price=100,
                volatility=-0.2,  # Negative volatility
                risk_free_rate=0.05,
                time_horizon=30/365
            )

    def test_distribution_moments(self, calculator):
        """Test calculation of distribution moments"""
        moments = calculator.calculate_distribution_moments()
        
        assert 'mean' in moments
        assert 'variance' in moments
        assert 'skewness' in moments
        assert 'kurtosis' in moments
        
        # Check moment relationships
        assert moments['variance'] >= 0
        assert np.isfinite(moments['skewness'])
        assert moments['kurtosis'] > 0

    def test_mixture_weights(self, calculator):
        """Test mixture distribution weights"""
        weights = [0.6, 0.4]
        params = [(0.2, 0.05), (0.4, 0.1)]  # [(vol1, drift1), (vol2, drift2)]
        
        mixture_dist = calculator.calculate_probability_density(
            distribution_type=DistributionType.MIXTURE,
            mixture_weights=weights,
            mixture_params=params
        )
        
        assert len(mixture_dist) > 0
        assert np.all(mixture_dist >= 0)
        assert np.isclose(np.sum(mixture_dist) * calculator.price_step, 1.0, rtol=1e-2)

    def test_time_scaling(self, calculator):
        """Test probability scaling with time"""
        short_time = calculator.calculate_probability_density(time_horizon=1/365)
        long_time = calculator.calculate_probability_density(time_horizon=252/365)
        
        # Variance should increase with time
        assert np.var(short_time) < np.var(long_time) 