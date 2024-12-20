import pytest
import numpy as np
import pandas as pd
from src.utils.volatility import VolatilityAnalyzer, VolatilitySurface, VolatilityRegime
from datetime import datetime

@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing"""
    return pd.DataFrame({
        'strike': [95, 100, 105] * 3,
        'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
        'option_type': ['call'] * 9,
        'underlying_price': [100] * 9
    })

@pytest.fixture
def sample_historical_data():
    """Fixture providing sample historical data for testing"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    return pd.DataFrame({
        'date': dates,
        'price': np.random.lognormal(mean=4.6, sigma=0.2, size=len(dates)),
        'historical_vol': np.random.normal(0.2, 0.05, size=len(dates))
    })

class TestVolatilityAnalyzer:
    """Unit tests for VolatilityAnalyzer class"""

    def test_initialization(self, sample_market_data):
        """Test proper initialization of VolatilityAnalyzer"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        assert analyzer.spot_price == 100
        assert len(analyzer.market_data) == 9

    def test_invalid_initialization(self):
        """Test initialization with invalid data"""
        invalid_data = pd.DataFrame({'strike': [100]})  # Missing required columns
        with pytest.raises(ValueError):
            VolatilityAnalyzer(invalid_data)

    def test_vol_surface_construction(self, sample_market_data):
        """Test volatility surface construction"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface(method='cubic')
        
        assert isinstance(surface, VolatilitySurface)
        assert surface.implied_vols.shape[1] == len(np.unique(sample_market_data['strike']))
        assert not np.any(np.isnan(surface.implied_vols))

    def test_regime_detection(self, sample_market_data, sample_historical_data):
        """Test volatility regime detection"""
        analyzer = VolatilityAnalyzer(sample_market_data, sample_historical_data)
        regime = analyzer.detect_volatility_regime(window=30)
        
        assert isinstance(regime, VolatilityRegime)
        assert regime.regime_type in ['low_vol', 'normal', 'high_vol']
        assert all(0 <= prob <= 1 for prob in regime.transition_probs.values())

    def test_volatility_forecasting(self, sample_market_data, sample_historical_data):
        """Test volatility forecasting"""
        analyzer = VolatilityAnalyzer(sample_market_data, sample_historical_data)
        forecast = analyzer.forecast_volatility(horizon=30)
        
        assert isinstance(forecast, dict)
        assert 'point_forecast' in forecast
        assert 'confidence_intervals' in forecast
        assert forecast['point_forecast'] > 0

    @pytest.mark.parametrize("method", ['cubic', 'linear', 'rbf', 'svi'])
    def test_surface_construction_methods(self, sample_market_data, method):
        """Test different surface construction methods"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface(method=method)
        assert isinstance(surface, VolatilitySurface)

class TestVolatilitySurface:
    """Unit tests for VolatilitySurface class"""

    def test_surface_properties(self, sample_market_data):
        """Test volatility surface properties"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface()
        
        assert hasattr(surface, 'quality_metrics')
        assert all(metric in surface.quality_metrics for metric in ['rmse', 'max_error'])
        assert surface.timestamp <= pd.Timestamp.now()

class TestVolatilityRegime:
    """Unit tests for VolatilityRegime class"""

    def test_regime_properties(self):
        """Test volatility regime properties"""
        metrics = {'vol_level': 0.2, 'vol_of_vol': 0.05}
        transition_probs = {'low_vol': 0.3, 'normal': 0.6, 'high_vol': 0.1}
        regime = VolatilityRegime('normal', metrics, transition_probs)
        
        assert regime.regime_type == 'normal'
        assert regime.metrics == metrics
        assert sum(regime.transition_probs.values()) == pytest.approx(1.0) 