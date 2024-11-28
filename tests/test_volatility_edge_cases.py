import pytest
import numpy as np
import pandas as pd
from src.utils.volatility import VolatilityAnalyzer
from datetime import datetime, timedelta

class TestVolatilityEdgeCases:
    """Test edge cases and boundary conditions for volatility analysis"""

    @pytest.fixture
    def extreme_market_data(self):
        """Generate market data with extreme scenarios"""
        return pd.DataFrame({
            # Deep ITM/OTM options
            'strike': [1, 100, 1000] * 3,
            'expiry': [0.01, 0.01, 0.01, 30, 30, 30, 0.5, 0.5, 0.5],  # Very short/long dated
            'implied_vol': [0.01, 0.2, 2.0, 0.15, 0.25, 0.35, 0.1, 0.2, 0.3],  # Extreme vols
            'option_type': ['call'] * 9,
            'underlying_price': [100] * 9
        })

    @pytest.fixture
    def sparse_market_data(self):
        """Generate sparse market data"""
        return pd.DataFrame({
            'strike': [90, 110],
            'expiry': [0.25, 1.0],
            'implied_vol': [0.2, 0.25],
            'option_type': ['call'] * 2,
            'underlying_price': [100] * 2
        })

    def test_extreme_strikes(self, extreme_market_data):
        """Test handling of extreme strike prices"""
        analyzer = VolatilityAnalyzer(extreme_market_data)
        surface = analyzer.construct_vol_surface()
        
        # Check extrapolation behavior
        assert not np.any(np.isnan(surface.implied_vols))
        assert not np.any(np.isinf(surface.implied_vols))
        assert np.all(surface.implied_vols > 0)

    def test_sparse_data(self, sparse_market_data):
        """Test behavior with sparse market data"""
        analyzer = VolatilityAnalyzer(sparse_market_data)
        
        # Test different interpolation methods
        for method in ['cubic', 'linear', 'rbf']:
            surface = analyzer.construct_vol_surface(method=method)
            assert surface.quality_metrics['interpolation_method'] == method
            assert 'sparsity_warning' in surface.quality_metrics

    def test_high_volatility_regime(self):
        """Test regime detection in high volatility environments"""
        # Generate high volatility historical data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        high_vol_data = pd.DataFrame({
            'date': dates,
            'price': np.random.lognormal(mean=4.6, sigma=0.8, size=len(dates)),  # High sigma
            'historical_vol': np.random.normal(0.8, 0.2, size=len(dates))  # High vol
        })
        
        analyzer = VolatilityAnalyzer(self.extreme_market_data(), high_vol_data)
        regime = analyzer.detect_volatility_regime()
        
        assert regime.regime_type == 'high_vol'
        assert regime.metrics['vol_level'] > 0.5

    def test_zero_dte_options(self):
        """Test handling of zero days to expiry"""
        zero_dte_data = pd.DataFrame({
            'strike': [95, 100, 105],
            'expiry': [0.0, 0.0, 0.0],  # Zero DTE
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(zero_dte_data)
        with pytest.raises(ValueError, match="Zero DTE options not supported"):
            analyzer.construct_vol_surface()

    def test_volatility_discontinuities(self):
        """Test handling of volatility surface discontinuities"""
        # Create data with volatility jumps
        strikes = np.linspace(90, 110, 21)
        expiries = [0.25, 0.5, 1.0]
        data = []
        for strike in strikes:
            for expiry in expiries:
                # Add discontinuity around ATM
                vol_jump = 0.1 if strike > 100 else 0.0
                data.append({
                    'strike': strike,
                    'expiry': expiry,
                    'implied_vol': 0.2 + vol_jump,
                    'option_type': 'call',
                    'underlying_price': 100
                })
        
        discontinuous_data = pd.DataFrame(data)
        analyzer = VolatilityAnalyzer(discontinuous_data)
        surface = analyzer.construct_vol_surface()
        
        assert 'discontinuity_detected' in surface.quality_metrics
        assert surface.quality_metrics['smoothness_score'] < 0.8 