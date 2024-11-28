import pytest
import numpy as np
from src.utils.greeks import Greeks, GreeksCalculator

class TestGreeks:
    """Test suite for Greeks calculations"""

    @pytest.fixture
    def sample_greeks(self):
        """Sample Greeks instance for testing"""
        return Greeks(
            delta=0.5,
            gamma=0.06,
            theta=-0.15,
            vega=0.2,
            rho=0.1
        )

    @pytest.fixture
    def calculator(self):
        """Sample GreeksCalculator instance"""
        return GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=0.2
        )

    def test_greeks_initialization(self, sample_greeks):
        """Test proper initialization of Greeks class"""
        assert sample_greeks.delta == 0.5
        assert sample_greeks.gamma == 0.06
        assert sample_greeks.theta == -0.15
        assert sample_greeks.vega == 0.2
        assert sample_greeks.rho == 0.1

    def test_calculator_initialization(self, calculator):
        """Test proper initialization of GreeksCalculator"""
        assert calculator.spot_price == 100
        assert calculator.strike == 100
        assert calculator.time_to_expiry == 0.5
        assert calculator.risk_free_rate == 0.05
        assert calculator.volatility == 0.2

    @pytest.mark.parametrize("option_type,expected_delta", [
        ('call', 0.5),  # ATM call should have ~0.5 delta
        ('put', -0.5)   # ATM put should have ~-0.5 delta
    ])
    def test_delta_calculation(self, calculator, option_type, expected_delta):
        """Test delta calculations for calls and puts"""
        greeks = calculator.calculate_greeks(option_type)
        assert abs(greeks.delta - expected_delta) < 0.1

    def test_gamma_positive(self, calculator):
        """Test that gamma is always positive"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        assert call_greeks.gamma > 0
        assert put_greeks.gamma > 0
        assert abs(call_greeks.gamma - put_greeks.gamma) < 1e-10

    def test_theta_signs(self, calculator):
        """Test that theta is typically negative"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        assert call_greeks.theta < 0
        # Deep ITM puts can have positive theta
        # but for ATM options, theta should be negative
        assert put_greeks.theta < 0

    def test_vega_positive(self, calculator):
        """Test that vega is always positive"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        assert call_greeks.vega > 0
        assert put_greeks.vega > 0
        assert abs(call_greeks.vega - put_greeks.vega) < 1e-10

    @pytest.mark.parametrize("spot,strike,expected_delta", [
        (110, 100, 0.7),  # ITM call
        (90, 100, 0.3),   # OTM call
        (100, 100, 0.5)   # ATM call
    ])
    def test_moneyness_effect(self, spot, strike, expected_delta):
        """Test effect of moneyness on Greeks"""
        calculator = GreeksCalculator(
            spot_price=spot,
            strike=strike,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=0.2
        )
        greeks = calculator.calculate_greeks('call')
        assert abs(greeks.delta - expected_delta) < 0.1

    def test_time_decay(self):
        """Test time decay effect on Greeks"""
        long_dated = GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        ).calculate_greeks('call')
        
        short_dated = GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=0.1,
            risk_free_rate=0.05,
            volatility=0.2
        ).calculate_greeks('call')
        
        assert abs(long_dated.gamma) < abs(short_dated.gamma)
        assert abs(long_dated.theta) < abs(short_dated.theta)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        with pytest.raises(ValueError):
            GreeksCalculator(
                spot_price=-100,  # Negative spot price
                strike=100,
                time_to_expiry=0.5,
                risk_free_rate=0.05,
                volatility=0.2
            )
        
        with pytest.raises(ValueError):
            GreeksCalculator(
                spot_price=100,
                strike=100,
                time_to_expiry=-0.5,  # Negative time
                risk_free_rate=0.05,
                volatility=0.2
            )
        
        with pytest.raises(ValueError):
            GreeksCalculator(
                spot_price=100,
                strike=100,
                time_to_expiry=0.5,
                risk_free_rate=0.05,
                volatility=-0.2  # Negative volatility
            )

    def test_extreme_values(self):
        """Test behavior with extreme input values"""
        calculator = GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=0.001,  # Very short dated
            risk_free_rate=0.05,
            volatility=0.2
        )
        greeks = calculator.calculate_greeks('call')
        
        assert greeks.delta in [0, 1]  # Should be binary
        assert abs(greeks.gamma) < 1e-3  # Should be near zero 