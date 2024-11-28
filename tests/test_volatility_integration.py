import pytest
import pandas as pd
import numpy as np
from src.utils.volatility import VolatilityAnalyzer
from src.utils.volatility_viz import VolatilityVisualizer
from src.utils.greeks import GreeksCalculator

class TestVolatilityIntegration:
    """Integration tests for volatility analysis components"""

    @pytest.fixture
    def setup_data(self):
        """Setup comprehensive test data"""
        # Market data
        strikes = np.linspace(90, 110, 21)
        expiries = [0.25, 0.5, 1.0]
        data = []
        for strike in strikes:
            for expiry in expiries:
                data.append({
                    'strike': strike,
                    'expiry': expiry,
                    'implied_vol': 0.2 + 0.01 * (100 - strike) / 100 + 0.02 * expiry,
                    'option_type': 'call',
                    'underlying_price': 100
                })
        market_data = pd.DataFrame(data)
        
        # Historical data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        historical_data = pd.DataFrame({
            'date': dates,
            'price': np.random.lognormal(mean=4.6, sigma=0.2, size=len(dates)),
            'historical_vol': np.random.normal(0.2, 0.05, size=len(dates))
        })
        
        return market_data, historical_data

    def test_end_to_end_analysis(self, setup_data):
        """Test complete volatility analysis workflow"""
        market_data, historical_data = setup_data
        
        # Initialize components
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        visualizer = VolatilityVisualizer()
        
        # Perform analysis
        surface = analyzer.construct_vol_surface(method='cubic')
        regime = analyzer.detect_volatility_regime()
        forecast = analyzer.forecast_volatility()
        
        # Create visualizations
        surface_plot = visualizer.plot_volatility_surface(surface)
        smile_plot = visualizer.plot_volatility_smile(
            surface.strikes,
            surface.implied_vols[0],
            surface.expiries[0]
        )
        
        # Verify results
        assert surface is not None
        assert regime.regime_type in ['low_vol', 'normal', 'high_vol']
        assert forecast['point_forecast'] > 0
        assert surface_plot is not None
        assert smile_plot is not None

    def test_greeks_integration(self, setup_data):
        """Test integration with Greeks calculations"""
        market_data, _ = setup_data
        analyzer = VolatilityAnalyzer(market_data)
        surface = analyzer.construct_vol_surface()
        
        # Calculate Greeks using surface volatilities
        greeks_calc = GreeksCalculator(
            spot_price=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=surface.implied_vols[len(surface.expiries)//2, len(surface.strikes)//2]
        )
        
        greeks = greeks_calc.calculate_greeks()
        assert all(hasattr(greeks, attr) for attr in ['delta', 'gamma', 'theta', 'vega', 'rho'])

    def test_regime_forecast_integration(self, setup_data):
        """Test integration between regime detection and forecasting"""
        market_data, historical_data = setup_data
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        
        # Detect regime and create forecast
        regime = analyzer.detect_volatility_regime()
        forecast = analyzer.forecast_volatility()
        
        # Verify regime impacts forecast
        assert 'regime_component' in forecast['decomposition']
        assert forecast['weights']['regime'] > 0

    def test_error_propagation(self, setup_data):
        """Test error handling across components"""
        market_data, historical_data = setup_data
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            analyzer.construct_vol_surface(method='invalid_method')
        
        with pytest.raises(ValueError):
            analyzer.forecast_volatility(horizon=-1) 