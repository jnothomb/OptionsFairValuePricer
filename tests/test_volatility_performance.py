import pytest
import numpy as np
import pandas as pd
import time
from src.utils.volatility import VolatilityAnalyzer

class TestVolatilityPerformance:
    """Performance testing for volatility analysis"""

    @pytest.fixture
    def large_market_data(self):
        """Generate large market dataset"""
        strikes = np.linspace(50, 150, 101)  # 101 strikes
        expiries = np.linspace(0.1, 5.0, 50)  # 50 expiries
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
        return pd.DataFrame(data)

    @pytest.fixture
    def large_historical_data(self):
        """Generate large historical dataset"""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'price': np.random.lognormal(mean=4.6, sigma=0.2, size=len(dates)),
            'historical_vol': np.random.normal(0.2, 0.05, size=len(dates))
        })

    def test_surface_construction_performance(self, large_market_data):
        """Test performance of surface construction"""
        analyzer = VolatilityAnalyzer(large_market_data)
        
        # Test different methods
        methods = ['cubic', 'linear', 'rbf', 'svi']
        performance_metrics = {}
        
        for method in methods:
            start_time = time.time()
            surface = analyzer.construct_vol_surface(method=method)
            end_time = time.time()
            
            performance_metrics[method] = {
                'execution_time': end_time - start_time,
                'memory_usage': surface.implied_vols.nbytes / 1024 / 1024  # MB
            }
        
        # Assert reasonable performance
        for method in methods:
            assert performance_metrics[method]['execution_time'] < 5.0  # Max 5 seconds
            assert performance_metrics[method]['memory_usage'] < 100  # Max 100MB

    def test_regime_detection_performance(self, large_market_data, large_historical_data):
        """Test performance of regime detection"""
        analyzer = VolatilityAnalyzer(large_market_data, large_historical_data)
        
        start_time = time.time()
        regime = analyzer.detect_volatility_regime(window=252)  # 1 year window
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 2.0  # Max 2 seconds

    def test_forecast_performance(self, large_market_data, large_historical_data):
        """Test performance of volatility forecasting"""
        analyzer = VolatilityAnalyzer(large_market_data, large_historical_data)
        
        start_time = time.time()
        forecast = analyzer.forecast_volatility(horizon=30)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 3.0  # Max 3 seconds

    @pytest.mark.benchmark
    def test_full_analysis_benchmark(self, large_market_data, large_historical_data):
        """Benchmark complete volatility analysis workflow"""
        analyzer = VolatilityAnalyzer(large_market_data, large_historical_data)
        
        def run_full_analysis():
            surface = analyzer.construct_vol_surface()
            regime = analyzer.detect_volatility_regime()
            forecast = analyzer.forecast_volatility()
            return surface, regime, forecast
        
        # Run multiple iterations
        iterations = 5
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            run_full_analysis()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        assert avg_time < 10.0  # Max 10 seconds average
        assert std_time < 1.0   # Max 1 second standard deviation 