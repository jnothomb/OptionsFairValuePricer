import pytest
import numpy as np
import pandas as pd
from src.utils.risk_manager import RiskManager, RiskThresholds, PositionRisk
from src.utils.greeks import Greeks

class TestRiskManager:
    """Test suite for Risk Manager functionality"""

    @pytest.fixture
    def risk_thresholds(self):
        """Sample risk thresholds for testing"""
        return RiskThresholds(
            max_position_delta=1.0,
            max_position_gamma=0.1,
            max_position_vega=0.5,
            max_position_theta=-0.2,
            max_position_size=100000,
            max_loss_threshold=5000
        )

    @pytest.fixture
    def sample_position(self):
        """Sample position data for testing"""
        return {
            'type': 'call',
            'quantity': 10,
            'strike': 100,
            'expiry': '2024-12-31',
            'underlying_price': 100,
            'option_price': 5.0,
            'implied_vol': 0.2
        }

    @pytest.fixture
    def risk_manager(self, risk_thresholds):
        """Initialize RiskManager with thresholds"""
        return RiskManager(risk_thresholds)

    def test_risk_manager_initialization(self, risk_manager, risk_thresholds):
        """Test proper initialization of RiskManager"""
        assert risk_manager.thresholds == risk_thresholds
        assert isinstance(risk_manager.positions, list)
        assert isinstance(risk_manager.risk_metrics, dict)

    def test_add_position(self, risk_manager, sample_position):
        """Test adding a new position"""
        position_id = risk_manager.add_position(sample_position)
        assert len(risk_manager.positions) == 1
        assert risk_manager.positions[0]['id'] == position_id
        assert risk_manager.positions[0]['type'] == 'call'

    def test_remove_position(self, risk_manager, sample_position):
        """Test removing a position"""
        position_id = risk_manager.add_position(sample_position)
        risk_manager.remove_position(position_id)
        assert len(risk_manager.positions) == 0

    def test_position_risk_calculation(self, risk_manager, sample_position):
        """Test calculation of position risk metrics"""
        position_id = risk_manager.add_position(sample_position)
        risk_metrics = risk_manager.calculate_position_risk(position_id)
        
        assert isinstance(risk_metrics, PositionRisk)
        assert hasattr(risk_metrics, 'delta')
        assert hasattr(risk_metrics, 'gamma')
        assert hasattr(risk_metrics, 'theta')
        assert hasattr(risk_metrics, 'vega')

    def test_portfolio_risk_calculation(self, risk_manager):
        """Test calculation of portfolio-wide risk metrics"""
        # Add multiple positions
        positions = [
            {
                'type': 'call',
                'quantity': 10,
                'strike': 100,
                'expiry': '2024-12-31',
                'underlying_price': 100,
                'option_price': 5.0,
                'implied_vol': 0.2
            },
            {
                'type': 'put',
                'quantity': -5,
                'strike': 95,
                'expiry': '2024-12-31',
                'underlying_price': 100,
                'option_price': 3.0,
                'implied_vol': 0.25
            }
        ]
        
        for position in positions:
            risk_manager.add_position(position)
        
        portfolio_risk = risk_manager.calculate_portfolio_risk()
        
        assert 'total_delta' in portfolio_risk
        assert 'total_gamma' in portfolio_risk
        assert 'total_theta' in portfolio_risk
        assert 'total_vega' in portfolio_risk
        assert 'total_exposure' in portfolio_risk

    def test_risk_threshold_violations(self, risk_manager):
        """Test detection of risk threshold violations"""
        # Add position that exceeds delta threshold
        large_position = {
            'type': 'call',
            'quantity': 100,  # Large position
            'strike': 100,
            'expiry': '2024-12-31',
            'underlying_price': 100,
            'option_price': 5.0,
            'implied_vol': 0.2
        }
        
        position_id = risk_manager.add_position(large_position)
        violations = risk_manager.check_risk_violations()
        
        assert len(violations) > 0
        assert any('delta' in v.lower() for v in violations)

    def test_risk_adjustments(self, risk_manager, sample_position):
        """Test suggested risk adjustments"""
        position_id = risk_manager.add_position(sample_position)
        adjustments = risk_manager.suggest_risk_adjustments()
        
        assert isinstance(adjustments, dict)
        assert 'suggested_trades' in adjustments
        assert 'risk_impact' in adjustments

    @pytest.mark.parametrize("risk_metric,threshold", [
        ('delta', 2.0),
        ('gamma', 0.2),
        ('vega', 1.0),
        ('theta', -0.4)
    ])
    def test_individual_risk_thresholds(self, risk_metric, threshold):
        """Test individual risk metric thresholds"""
        custom_thresholds = RiskThresholds(
            max_position_delta=2.0,
            max_position_gamma=0.2,
            max_position_vega=1.0,
            max_position_theta=-0.4,
            max_position_size=100000,
            max_loss_threshold=5000
        )
        
        risk_manager = RiskManager(custom_thresholds)
        assert getattr(risk_manager.thresholds, f'max_position_{risk_metric}') == threshold

    def test_stress_testing(self, risk_manager, sample_position):
        """Test stress testing scenarios"""
        position_id = risk_manager.add_position(sample_position)
        
        stress_scenarios = {
            'market_crash': {'price_change': -0.2, 'vol_change': 0.5},
            'rally': {'price_change': 0.2, 'vol_change': -0.2},
            'vol_spike': {'price_change': 0, 'vol_change': 1.0}
        }
        
        stress_results = risk_manager.run_stress_tests(stress_scenarios)
        
        assert isinstance(stress_results, dict)
        assert all(scenario in stress_results for scenario in stress_scenarios)
        assert all('pnl_impact' in result for result in stress_results.values())

    def test_risk_reporting(self, risk_manager, sample_position):
        """Test risk reporting functionality"""
        position_id = risk_manager.add_position(sample_position)
        report = risk_manager.generate_risk_report()
        
        assert isinstance(report, dict)
        assert 'portfolio_risk' in report
        assert 'risk_violations' in report
        assert 'stress_test_results' in report
        assert 'suggested_actions' in report 