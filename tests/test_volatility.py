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
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'date': dates,
        'price': 100 * np.exp(np.random.normal(0, 0.02, size=len(dates))),
        'historical_vol': np.abs(np.random.normal(0.2, 0.05, size=len(dates)))
    })

class TestVolatilityAnalyzer:
    """Unit tests for VolatilityAnalyzer class"""

    def test_initialization(self, sample_market_data):
        """Test proper initialization of VolatilityAnalyzer"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        assert analyzer.spot_price == 100
        assert len(analyzer.market_data) == 9  # Fixed: Correct number of rows

    def test_invalid_initialization(self):
        """Test initialization with invalid data"""
        invalid_data = pd.DataFrame({
            'strike': [100],
            'expiry': [0.5],
            'implied_vol': [0.2],
            'option_type': ['call']
        })  # Missing underlying_price
        with pytest.raises(ValueError, match="missing required columns"):
            VolatilityAnalyzer(invalid_data)

    @pytest.mark.parametrize("bad_data,expected_error", [
        (pd.DataFrame({
            'strike': [-100, 100],
            'expiry': [0.1, 0.1],
            'implied_vol': [0.2, 0.2],
            'option_type': ['call', 'call'],
            'underlying_price': [100, 100]
        }), "Strike prices must be positive"),
        (pd.DataFrame({
            'strike': [100, 100],
            'expiry': [-1, 1],
            'implied_vol': [0.2, 0.2],
            'option_type': ['call', 'call'],
            'underlying_price': [100, 100]
        }), "Expiry times must be positive"),
        (pd.DataFrame({
            'strike': [100, 100],
            'expiry': [0.1, 0.1],
            'implied_vol': [-0.2, 0.2],
            'option_type': ['call', 'call'],
            'underlying_price': [100, 100]
        }), "Implied volatilities must be positive")
    ])
    def test_data_validation(self, bad_data, expected_error):
        """Test handling of invalid input data"""
        with pytest.raises(ValueError, match=expected_error):
            VolatilityAnalyzer(bad_data)

    def test_regime_detection(self, sample_market_data, sample_historical_data):
        """Test volatility regime detection"""
        analyzer = VolatilityAnalyzer(sample_market_data, sample_historical_data)
        regime = analyzer.detect_volatility_regime(window=30)
        
        assert isinstance(regime, VolatilityRegime)
        assert regime.regime_type in ['low_vol', 'normal', 'high_vol']
        assert all(0 <= prob <= 1 for prob in regime.transition_probs.values())
        assert 'regime_stability' in regime.metrics

    def test_volatility_forecasting(self, sample_market_data, sample_historical_data):
        """Test volatility forecasting"""
        analyzer = VolatilityAnalyzer(sample_market_data, sample_historical_data)
        forecast = analyzer.forecast_volatility(horizon=30)
        
        assert isinstance(forecast, dict)
        assert 'point_forecast' in forecast
        assert 'confidence_intervals' in forecast
        assert 0 < forecast['point_forecast'] < 1.0  # Reasonable vol range
        
        # Test with no historical data
        analyzer_no_hist = VolatilityAnalyzer(sample_market_data)
        forecast_no_hist = analyzer_no_hist.forecast_volatility(horizon=30)
        assert isinstance(forecast_no_hist, dict)
        assert 0 < forecast_no_hist['point_forecast'] < 1.0

    def test_volatility_forecast_fallback(self, sample_market_data):
        """Test volatility forecast fallback mechanism"""
        analyzer = VolatilityAnalyzer(sample_market_data)  # No historical data
        
        # Should fall back to implied vol average
        forecast = analyzer._historical_vol_forecast(horizon=30)
        assert isinstance(forecast, float)
        assert abs(forecast - sample_market_data['implied_vol'].mean()) < 1e-6

    def test_garch_data_validation(self, sample_historical_data):
        """Test GARCH data validation"""
        market_data = pd.DataFrame({
            'strike': [100],
            'expiry': [0.5],
            'implied_vol': [0.2],
            'option_type': ['call'],
            'underlying_price': [100]
        })
        
        analyzer = VolatilityAnalyzer(market_data, sample_historical_data)
        
        # Test with numpy array
        returns = np.random.normal(0, 0.01, 252)  # One year of daily returns
        assert analyzer._validate_garch_data(returns)
        
        # Test with pandas Series
        returns_series = pd.Series(returns)
        assert analyzer._validate_garch_data(returns_series)
        
        # Test insufficient data
        short_returns = np.random.normal(0, 0.01, 50)
        assert not analyzer._validate_garch_data(short_returns)
        
        # Test invalid input
        assert not analyzer._validate_garch_data([1, 2, 3])
        
        # Test zero variance
        zero_var_returns = np.zeros(252)
        assert not analyzer._validate_garch_data(zero_var_returns)

    def test_vol_surface_construction(self, sample_market_data):
        """Test volatility surface construction"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface(method='cubic')
        
        assert isinstance(surface, VolatilitySurface)
        assert surface.implied_vols.shape[1] == len(np.unique(sample_market_data['strike']))
        assert not np.any(np.isnan(surface.implied_vols))

    @pytest.mark.parametrize("method", ['cubic', 'linear', 'rbf', 'svi'])
    def test_surface_construction_methods(self, sample_market_data, method):
        """Test different surface construction methods"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface(method=method)
        assert isinstance(surface, VolatilitySurface)

    def test_surface_arbitrage_free(self, sample_market_data):
        """Test no-arbitrage conditions in vol surface"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        surface = analyzer.construct_vol_surface()
        
        # Calendar spread arbitrage
        calendar_spreads = np.diff(surface.implied_vols, axis=1)
        assert np.all(calendar_spreads >= 0), "Calendar arbitrage detected"
        
        # Butterfly arbitrage
        butterfly_spreads = np.diff(surface.implied_vols, axis=0, n=2)
        assert np.all(butterfly_spreads >= 0), "Butterfly arbitrage detected"

    @pytest.mark.parametrize("vol_level,expected_regime", [
        (0.1, 'low_vol'),
        (0.2, 'normal'),
        (0.4, 'high_vol')
    ])
    def test_regime_classification(self, sample_market_data, vol_level, expected_regime):
        """Test regime classification under different vol conditions"""
        historical_data = self._generate_historical_data(mean_vol=vol_level)
        analyzer = VolatilityAnalyzer(sample_market_data, historical_data)
        regime = analyzer.detect_volatility_regime()
        assert regime.regime_type == expected_regime

    def _generate_historical_data(self, mean_vol: float) -> pd.DataFrame:
        """Generate historical data with specified mean volatility"""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'price': 100 * np.exp(np.random.normal(0, mean_vol/np.sqrt(252), len(dates))),
            'historical_vol': np.random.normal(mean_vol, 0.02, len(dates))
        })

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

class TestVolatilitySurfaceStability:
    """Test suite for volatility surface stability"""
    
    def test_surface_dimension_validation(self, sample_market_data):
        """Test surface dimension validation"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        
        # Test insufficient strikes
        with pytest.raises(ValueError, match="Insufficient data points"):
            strikes = np.array([100])
            expiries = np.array([0.25, 0.5])
            vols = np.zeros((1, 2))
            analyzer._validate_surface_dimensions(strikes, expiries, vols)
        
        # Test shape mismatch
        with pytest.raises(ValueError, match="Surface shape mismatch"):
            strikes = np.array([95, 100, 105])
            expiries = np.array([0.25, 0.5])
            vols = np.zeros((3, 3))  # Wrong shape
            analyzer._validate_surface_dimensions(strikes, expiries, vols)
    
    def test_surface_stabilization(self, sample_market_data):
        """Test surface numerical stabilization"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        
        # Create unstable surface
        unstable_vols = np.array([
            [0.2, 0.18, 0.16],  # Calendar arbitrage
            [-0.1, 0.2, 0.3],   # Negative vol
            [0.3, 0.25, 0.2]    # More calendar arbitrage
        ])
        
        stabilized = analyzer._stabilize_surface_calculation(unstable_vols)
        
        # Test bounds
        assert np.all(stabilized >= 0.001)
        assert np.all(stabilized <= 5.0)
        
        # Test calendar spread arbitrage
        calendar_spreads = np.diff(stabilized, axis=1)
        assert np.all(calendar_spreads >= 0)
        
        # Test butterfly arbitrage
        butterfly_spreads = (stabilized[:-2, :] + stabilized[2:, :]) / 2 - stabilized[1:-1, :]
        assert np.all(butterfly_spreads >= -1e-10)  # Allow for numerical precision
    
    def test_garch_data_validation(self, sample_historical_data):
        """Test GARCH data validation"""
        market_data = pd.DataFrame({
            'strike': [100],
            'expiry': [0.5],
            'implied_vol': [0.2],
            'option_type': ['call'],
            'underlying_price': [100]
        })
        
        analyzer = VolatilityAnalyzer(market_data, sample_historical_data)
        
        # Test with valid data
        returns = np.random.normal(0, 0.01, 252)
        assert analyzer._validate_garch_data(returns)
        
        # Test with invalid data
        assert not analyzer._validate_garch_data(np.zeros(252))
    
    def test_volatility_forecast_fallback(self, sample_market_data):
        """Test volatility forecast fallback mechanism"""
        analyzer = VolatilityAnalyzer(sample_market_data)
        
        # Should fall back to simple vol without historical data
        forecast = analyzer._historical_vol_forecast(horizon=30)
        assert isinstance(forecast, float)
        assert 0.01 <= forecast <= 1.0  # Reasonable vol range 