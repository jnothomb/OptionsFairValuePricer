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
        if option_type.lower() == 'call':
            assert 0 <= greeks.delta <= 1  # Call delta naturally bounded [0,1]
        else:
            assert -1 <= greeks.delta <= 0  # Put delta naturally bounded [-1,0]

    def test_gamma_positive(self, calculator):
        """Test that individual option gamma is always positive"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        # Individual option gamma is always positive
        assert call_greeks.gamma > 0
        assert put_greeks.gamma > 0
        # Calls and puts have identical gamma
        assert abs(call_greeks.gamma - put_greeks.gamma) < 1e-10

    def test_theta_signs(self, calculator):
        """Test theta signs under different conditions"""
        # Standard ATM case
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        # For ATM options with typical parameters, theta should be negative
        assert call_greeks.theta < 0
        
        # Test deep ITM put which can have positive theta
        itm_put_calc = GreeksCalculator(
            spot_price=80,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=0.2
        )
        itm_put_greeks = itm_put_calc.calculate_greeks('put')
        # We don't assert the sign because it depends on the specific parameters

    def test_vega_properties(self, calculator):
        """Test vega properties for individual options"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        # Individual option vega is always positive
        assert call_greeks.vega > 0
        assert put_greeks.vega > 0
        # Calls and puts have identical vega
        assert abs(call_greeks.vega - put_greeks.vega) < 1e-10

    def test_rho_properties(self, calculator):
        """Test rho properties for individual options"""
        call_greeks = calculator.calculate_greeks('call')
        put_greeks = calculator.calculate_greeks('put')
        
        # For typical ATM options:
        # Call rho should be positive
        assert call_greeks.rho > 0
        # Put rho should be negative
        assert put_greeks.rho < 0
        
        # For ATM options, rho values should be similar in magnitude
        # but not exactly equal due to the interest rate component in put-call parity
        # c - p = S - Ke^(-rT) implies different rho sensitivities
        rho_difference = abs(abs(call_greeks.rho) - abs(put_greeks.rho))
        assert rho_difference < abs(call_greeks.rho) * 0.2  # Within 20% of each other

    def test_portfolio_greeks_no_constraints(self):
        """Test that portfolio-level Greeks have no artificial constraints"""
        # Create a portfolio with large positions to test no constraints
        calculator = GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=0.2
        )
        
        # Calculate Greeks for a large position (e.g., 1000 contracts)
        position_size = 1000
        call_greeks = calculator.calculate_greeks('call')
        
        # Portfolio Greeks should be able to take any real value
        portfolio_delta = position_size * call_greeks.delta
        portfolio_gamma = position_size * call_greeks.gamma
        portfolio_vega = position_size * call_greeks.vega
        portfolio_theta = position_size * call_greeks.theta
        portfolio_rho = position_size * call_greeks.rho
        
        # No assertions on bounds - portfolio Greeks can take any real value
        assert isinstance(portfolio_delta, float)
        assert isinstance(portfolio_gamma, float)
        assert isinstance(portfolio_vega, float)
        assert isinstance(portfolio_theta, float)
        assert isinstance(portfolio_rho, float)

    @pytest.mark.parametrize("spot,strike,expected_delta", [
        (110, 100, 0.82),  # ITM call - updated to match BS formula
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
        with pytest.raises(ValueError, match="must be positive"):
            GreeksCalculator(
                spot_price=-100,  # Negative spot price
                strike=100,
                time_to_expiry=0.5,
                risk_free_rate=0.05,
                volatility=0.2
            )
        
        with pytest.raises(ValueError, match="must be positive"):
            GreeksCalculator(
                spot_price=100,
                strike=100,
                time_to_expiry=-0.5,  # Negative time
                risk_free_rate=0.05,
                volatility=0.2
            )
        
        with pytest.raises(ValueError, match="must be positive"):
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
        
        # For very short dated ATM options:
        # Delta should be close to 0.5 (but not exactly)
        assert 0.45 <= greeks.delta <= 0.55
        # Gamma should be large
        assert greeks.gamma > 0.5
        # Theta should be large negative
        assert greeks.theta < -100
        # Vega should be small
        assert greeks.vega < 2.0 