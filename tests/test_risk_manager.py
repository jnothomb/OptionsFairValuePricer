import pytest
import numpy as np
import pandas as pd
from src.utils.risk_manager import RiskManager, RiskMetrics, RiskThresholds
from src.utils.volatility import VolatilityAnalyzer

@pytest.fixture
def sample_positions():
    """Fixture providing sample options positions"""
    return pd.DataFrame({
        'strike': [95.0, 100.0, 105.0],
        'expiry': [0.25, 0.25, 0.25],
        'position': [1.0, -2.0, 1.0],  # Long strangle, short straddle
        'option_type': ['call', 'call', 'put']
    })

@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for volatility analyzer"""
    return pd.DataFrame({
        'strike': [95, 100, 105] * 3,
        'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
        'option_type': ['call'] * 9,
        'underlying_price': [100] * 9
    })

@pytest.fixture
def risk_manager(sample_positions, sample_market_data):
    """Fixture providing initialized risk manager"""
    vol_analyzer = VolatilityAnalyzer(sample_market_data)
    return RiskManager(sample_positions, vol_analyzer)

class TestRiskManager:
    """Test suite for RiskManager class"""

    def test_initialization(self, sample_positions, sample_market_data):
        """Test proper initialization of RiskManager"""
        vol_analyzer = VolatilityAnalyzer(sample_market_data)
        manager = RiskManager(sample_positions, vol_analyzer)
        
        assert manager.spot_price == 100.0
        assert len(manager.positions) == 3
        assert manager.risk_free_rate == pytest.approx(0.02)

    def test_invalid_initialization(self):
        """Test initialization with invalid data"""
        vol_analyzer = VolatilityAnalyzer(pd.DataFrame({
            'strike': [100],
            'expiry': [0.5],
            'implied_vol': [0.2],
            'option_type': ['call'],
            'underlying_price': [100]
        }))
        
        invalid_positions = pd.DataFrame({
            'strike': [100],  # Missing required columns
            'expiry': [0.5]
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            RiskManager(invalid_positions, vol_analyzer)

    def test_portfolio_greeks(self, risk_manager):
        """Test portfolio Greeks calculation"""
        greeks = risk_manager.calculate_portfolio_greeks()
        
        assert isinstance(greeks, RiskMetrics)
        
        # Test individual position deltas (should be naturally bounded)
        for _, pos in risk_manager.positions.iterrows():
            single_pos_greeks = risk_manager._calculate_position_greeks(pos)
            base_delta = single_pos_greeks.delta / pos['position']  # Remove position effect
            if pos['option_type'].lower() == 'call':
                assert 0 <= base_delta <= 1  # Call delta naturally bounded [0,1]
            else:
                assert -1 <= base_delta <= 0  # Put delta naturally bounded [-1,0]
        
        # Test individual position gamma (should be positive)
        for _, pos in risk_manager.positions.iterrows():
            single_pos_greeks = risk_manager._calculate_position_greeks(pos)
            base_gamma = single_pos_greeks.gamma / pos['position']  # Remove position effect
            assert base_gamma > 0  # Base gamma should be positive
            
        # Portfolio Greeks can take any real value
        assert isinstance(greeks.delta, float)
        assert isinstance(greeks.gamma, float)
        assert isinstance(greeks.vega, float)
        assert isinstance(greeks.theta, float)
        assert isinstance(greeks.rho, float)
        assert np.isfinite(greeks.delta)
        assert np.isfinite(greeks.gamma)
        assert np.isfinite(greeks.vega)
        assert np.isfinite(greeks.theta)
        assert np.isfinite(greeks.rho)

    def test_value_at_risk(self, risk_manager):
        """Test VaR calculation"""
        var_99 = risk_manager.calculate_var(confidence=0.99)
        var_95 = risk_manager.calculate_var(confidence=0.95)
        
        assert var_99 > var_95  # Higher confidence = larger VaR
        assert isinstance(var_99, float)
        assert var_99 > 0  # VaR should be positive for long options

    def test_expected_shortfall(self, risk_manager):
        """Test Expected Shortfall calculation"""
        es_99 = risk_manager.calculate_expected_shortfall(confidence=0.99)
        var_99 = risk_manager.calculate_var(confidence=0.99)
        
        assert es_99 > var_99  # ES should be larger than VaR
        assert isinstance(es_99, float)
        assert es_99 > 0

    def test_hedge_recommendations(self, risk_manager):
        """Test hedging recommendations"""
        recommendations = risk_manager.generate_hedge_recommendations()
        
        assert isinstance(recommendations, dict)
        assert all(key in recommendations for key in ['delta_hedge', 'vega_hedge', 'gamma_hedge'])
        assert all(isinstance(v, float) for v in recommendations.values())

    def test_scenario_analysis(self, risk_manager):
        """Test scenario analysis"""
        spot_changes = [-0.1, 0.0, 0.1]
        vol_changes = [-0.2, 0.0, 0.2]
        
        results = risk_manager.run_scenario_analysis(spot_changes, vol_changes)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(spot_changes) * len(vol_changes)
        assert all(col in results.columns for col in ['spot_change', 'vol_change', 'pnl_estimate'])

    def test_risk_limits(self, risk_manager):
        """Test risk limits monitoring"""
        # Use large limits to test that portfolio Greeks aren't artificially constrained
        limits = {
            'delta': 1000.0,  # Can be exceeded by large positions
            'vega': 10000.0,
            'var_99': 1000000.0
        }
        
        breaches = risk_manager.check_risk_limits(limits)
        
        assert isinstance(breaches, dict)
        assert all(isinstance(v, bool) for v in breaches.values())
        assert all(key in breaches for key in limits.keys())

    def test_stress_testing(self, risk_manager):
        """Test stress testing functionality"""
        scenarios = [
            {'spot_shock': 0.1, 'vol_shock': 0.2, 'rate_shock': 50},
            {'spot_shock': -0.1, 'vol_shock': -0.2, 'rate_shock': -50}
        ]
        
        results = risk_manager.calculate_stress_test(scenarios)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(scenarios)
        assert all(col in results.columns for col in 
                  ['total_pnl', 'delta_pnl', 'gamma_pnl', 'vega_pnl', 'theta_pnl', 'rho_pnl'])

    def test_sensitivity_analysis(self, risk_manager):
        """Test portfolio sensitivity analysis"""
        shock_range = [-0.1, 0.0, 0.1]
        
        for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
            results = risk_manager.calculate_portfolio_sensitivity(greek, shock_range)
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) == len(shock_range)
            assert all(col in results.columns for col in ['shock', 'pnl'])

    def test_margin_requirements(self, risk_manager):
        """Test margin requirements calculation"""
        margin_params = {
            'im_multiplier': 1.5,
            'mm_ratio': 0.75
        }
        
        requirements = risk_manager.calculate_margin_requirements(margin_params)
        
        assert isinstance(requirements, dict)
        assert all(key in requirements for key in 
                  ['initial_margin', 'maintenance_margin', 'exposure_component'])
        assert requirements['maintenance_margin'] < requirements['initial_margin']

    def test_risk_report(self, risk_manager):
        """Test risk report generation"""
        report = risk_manager.generate_risk_report()
        
        assert isinstance(report, dict)
        assert all(key in report for key in 
                  ['portfolio_greeks', 'risk_metrics', 'margin_requirements', 'hedge_recommendations'])
        assert isinstance(report['timestamp'], pd.Timestamp)

    def test_extreme_positions(self, sample_market_data):
        """Test handling of extreme positions"""
        extreme_positions = pd.DataFrame({
            'strike': [100.0],
            'expiry': [0.01],  # Very short dated
            'position': [100.0],  # Large position
            'option_type': ['call']
        })
        
        vol_analyzer = VolatilityAnalyzer(sample_market_data)
        manager = RiskManager(extreme_positions, vol_analyzer)
        
        # Test that calculations don't break with extreme values
        greeks = manager.calculate_portfolio_greeks()
        var = manager.calculate_var()
        es = manager.calculate_expected_shortfall()
        
        assert np.isfinite(greeks.delta)
        assert np.isfinite(var)
        assert np.isfinite(es)

    def test_portfolio_aggregation(self, sample_market_data):
        """Test portfolio aggregation with offsetting positions"""
        offsetting_positions = pd.DataFrame({
            'strike': [100.0, 100.0],
            'expiry': [0.25, 0.25],
            'position': [1.0, -1.0],  # Offsetting positions
            'option_type': ['call', 'call']
        })
        
        vol_analyzer = VolatilityAnalyzer(sample_market_data)
        manager = RiskManager(offsetting_positions, vol_analyzer)
        greeks = manager.calculate_portfolio_greeks()
        
        assert abs(greeks.delta) < 1e-10  # Should be close to zero
        assert abs(greeks.gamma) < 1e-10
        assert abs(greeks.vega) < 1e-10