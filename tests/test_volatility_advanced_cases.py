import pytest
import numpy as np
import pandas as pd
from src.utils.volatility import VolatilityAnalyzer
from datetime import datetime, timedelta

class TestVolatilityAdvancedCases:
    """Advanced test cases for volatility analysis"""

    @pytest.fixture
    def malformed_historical_data(self):
        """Generate various malformed historical datasets"""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        return {
            'gaps_in_data': pd.DataFrame({
                'date': list(dates[::2]),  # Skip every other day
                'price': np.random.lognormal(0, 0.2, size=len(dates[::2])),
                'historical_vol': np.random.normal(0.2, 0.05, size=len(dates[::2]))
            }),
            
            'non_trading_days': pd.DataFrame({
                'date': dates,
                'price': [np.nan if d.weekday() in [5, 6] else 100 for d in dates],
                'historical_vol': [np.nan if d.weekday() in [5, 6] else 0.2 for d in dates]
            }),
            
            'extreme_price_changes': pd.DataFrame({
                'date': dates,
                'price': [100] * len(dates),
                'historical_vol': [0.2] * len(dates)
            }).assign(
                price=lambda x: x['price'] * (1 + np.random.normal(0, 0.5, size=len(x)))
            ),
            
            'inconsistent_frequency': pd.DataFrame({
                'date': list(dates) + [dates[-1] + timedelta(hours=12)],
                'price': np.random.lognormal(0, 0.2, size=len(dates) + 1),
                'historical_vol': np.random.normal(0.2, 0.05, size=len(dates) + 1)
            })
        }

    def test_historical_data_gaps(self, malformed_historical_data):
        """Test handling of gaps in historical data"""
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(
            market_data, 
            malformed_historical_data['gaps_in_data']
        )
        
        # Should handle gaps gracefully
        regime = analyzer.detect_volatility_regime()
        assert regime is not None
        assert 'data_completeness' in regime.metrics

    def test_non_trading_days(self, malformed_historical_data):
        """Test handling of non-trading days in historical data"""
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(
            market_data, 
            malformed_historical_data['non_trading_days']
        )
        
        forecast = analyzer.forecast_volatility()
        assert 'trading_days_only' in forecast['diagnostics']

    @pytest.mark.parametrize("window", [5, 10, 21, 63, 252])
    def test_different_historical_windows(self, window, malformed_historical_data):
        """Test different historical data window sizes"""
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(market_data, malformed_historical_data['gaps_in_data'])
        regime = analyzer.detect_volatility_regime(window=window)
        assert regime is not None
        assert regime.metrics['window_size'] == window

    def test_extreme_price_changes(self, malformed_historical_data):
        """Test handling of extreme price changes in historical data"""
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(
            market_data, 
            malformed_historical_data['extreme_price_changes']
        )
        
        forecast = analyzer.forecast_volatility()
        assert 'extreme_events_detected' in forecast['diagnostics']

    def test_inconsistent_frequency(self, malformed_historical_data):
        """Test handling of inconsistent data frequency"""
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        with pytest.warns(UserWarning, match="Inconsistent data frequency detected"):
            analyzer = VolatilityAnalyzer(
                market_data, 
                malformed_historical_data['inconsistent_frequency']
            )

    @pytest.mark.parametrize("combination", [
        ('high_vol', 'low_strikes'),
        ('low_vol', 'high_strikes'),
        ('normal_vol', 'extreme_strikes')
    ])
    def test_invalid_combinations(self, combination):
        """Test invalid combinations of market data and historical data"""
        vol_scenario, strike_scenario = combination
        
        # Create market data based on strike scenario
        if strike_scenario == 'low_strikes':
            strikes = [1, 2, 3]
        elif strike_scenario == 'high_strikes':
            strikes = [1000, 2000, 3000]
        else:
            strikes = [1, 100, 1000]
            
        market_data = pd.DataFrame({
            'strike': strikes,
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        # Create historical data based on vol scenario
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        if vol_scenario == 'high_vol':
            hist_vol = np.random.normal(0.8, 0.2, size=len(dates))
        elif vol_scenario == 'low_vol':
            hist_vol = np.random.normal(0.05, 0.01, size=len(dates))
        else:
            hist_vol = np.random.normal(0.2, 0.05, size=len(dates))
            
        historical_data = pd.DataFrame({
            'date': dates,
            'price': np.random.lognormal(0, 0.2, size=len(dates)),
            'historical_vol': hist_vol
        })
        
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        
        # Check for warnings or adjustments in the analysis
        with pytest.warns(UserWarning):
            surface = analyzer.construct_vol_surface()
            assert 'data_consistency_warning' in surface.quality_metrics

    def test_calendar_effects(self):
        """Test handling of calendar effects in volatility"""
        # Generate data with known calendar effects
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        historical_data = pd.DataFrame({
            'date': dates,
            'price': np.random.lognormal(0, 0.2, size=len(dates)),
            'historical_vol': np.random.normal(0.2, 0.05, size=len(dates))
        })
        
        # Add seasonal effects
        historical_data['historical_vol'] += np.sin(
            np.pi * 2 * np.arange(len(dates)) / 252
        ) * 0.05
        
        market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(market_data, historical_data)
        forecast = analyzer.forecast_volatility()
        
        assert 'seasonality_detected' in forecast['diagnostics'] 