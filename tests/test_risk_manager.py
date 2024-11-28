import pytest
import numpy as np
from typing import Dict, List, Any
from src.utils.risk_manager import RiskManager, RiskThresholds

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
        
        assert isinstance(risk_metrics, dict)
        assert 'delta' in risk_metrics
        assert 'gamma' in risk_metrics
        assert 'theta' in risk_metrics
        assert 'vega' in risk_metrics

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

    @pytest.fixture
    def risk_manager(self):
        """Initialize RiskManager with sample thresholds and portfolio value."""
        thresholds = RiskThresholds(
            max_position_delta=1.0,
            max_position_gamma=0.1,
            max_position_vega=0.5,
            max_position_theta=-0.2,
            max_position_size=100000,
            max_loss_threshold=5000
        )
        manager = RiskManager(thresholds)
        manager.portfolio_value = 100000  # Example portfolio value
        return manager

    def test_calculate_position_size(self, risk_manager):
        """Test position sizing calculation."""
        account_balance = 10000
        risk_per_trade = 200
        entry_price = 50
        stop_loss_price = 45
        
        position_size = risk_manager.calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss_price)
        assert position_size == 40  # (200 / (50 - 45))

    def test_calculate_margin_requirement(self, risk_manager):
        """Test margin requirement calculation."""
        position_value = 50000
        margin_rate = 0.2
        
        margin_requirement = risk_manager.calculate_margin_requirement(position_value, margin_rate)
        assert margin_requirement == 10000  # (50000 * 0.2)

    def test_calculate_var(self, risk_manager):
        """Test Value at Risk (VaR) calculation."""
        portfolio_returns = np.array([-0.02, -0.01, 0.01, 0.03, -0.04])
        var = risk_manager.calculate_var(portfolio_returns, confidence_level=0.95)
        assert var == -0.04  # Expected VaR at 95% confidence level

    def test_run_stress_tests(self, risk_manager):
        """Test stress testing scenarios."""
        risk_manager.portfolio_value = 100000  # Set portfolio value for testing
        scenarios = {
            'market_crash': {'price_change': -0.2},
            'rally': {'price_change': 0.2}
        }
        
        results = risk_manager.run_stress_tests(scenarios)
        
        assert 'market_crash' in results
        assert 'rally' in results
        assert results['market_crash']['simulated_value'] == 80000  # 100000 * (1 - 0.2)
        assert results['rally']['simulated_value'] == 120000  # 100000 * (1 + 0.2)

    def test_check_position_risks(self, risk_manager):
        """Test the check_position_risks method."""
        # Assuming you have a method to add positions to the risk manager
        # Add a sample position for testing
        position_id = risk_manager.add_position({
            'type': 'call',
            'quantity': 10,
            'strike': 100,
            'expiry': '2024-12-31',
            'underlying_price': 100,
            'option_price': 5.0,
            'implied_vol': 0.2,
            'delta': 0.5,  # Example delta value
            'gamma': 0.1,  # Example gamma value
            'theta': -0.02,  # Example theta value
            'vega': 0.2,  # Example vega value
            'value': 5000  # Example position value
        })
        
        # Define greeks and position value for the test
        greeks = {
            'delta': 0.5,
            'gamma': 0.1,
            'theta': -0.02,
            'vega': 0.2
        }
        position_value = 5000  # Example position value

        # Check for risks
        risks = risk_manager.check_position_risks(greeks, position_value)
        assert isinstance(risks, dict)  # Assuming it returns a dictionary of risks
        assert 'delta' in risks  # Check for specific risk metrics

    def add_position(self, position: Dict[str, Any]) -> int:
        """Add a new position and return its ID."""
        position_id = len(self.positions) + 1  # Simple ID generation
        position['id'] = position_id  # Add ID to the position
        # Ensure the position has necessary risk metrics
        position['delta'] = position.get('delta', 0.0)  # Default to 0 if not provided
        position['gamma'] = position.get('gamma', 0.0)
        position['theta'] = position.get('theta', 0.0)
        position['vega'] = position.get('vega', 0.0)
        position['value'] = position.get('value', 0.0)  # Ensure value is set
        self.positions.append(position)  # Add position to the list
        return position_id

    def calculate_position_risk(self, position_id: int) -> Dict[str, float]:
        """Calculate risk metrics for a specific position."""
        position = next((pos for pos in self.positions if pos['id'] == position_id), None)
        if position is None:
            raise ValueError("Position not found.")
        
        # Return the risk metrics based on the position's attributes
        return {
            'delta': position.get('delta', 0.0),
            'gamma': position.get('gamma', 0.0),
            'theta': position.get('theta', 0.0),
            'vega': position.get('vega', 0.0)
        }

    def check_risk_violations(self) -> List[str]:
        """Check for risk threshold violations."""
        violations = []
        for position in self.positions:
            if 'delta' in position and abs(position['delta']) > self.thresholds.max_position_delta:
                violations.append(f"Position {position['id']} exceeds delta threshold.")
            if 'gamma' in position and abs(position['gamma']) > self.thresholds.max_position_gamma:
                violations.append(f"Position {position['id']} exceeds gamma threshold.")
            if 'theta' in position and position['theta'] < self.thresholds.max_position_theta:
                violations.append(f"Position {position['id']} exceeds theta threshold.")
            if 'vega' in position and abs(position['vega']) > self.thresholds.max_position_vega:
                violations.append(f"Position {position['id']} exceeds vega threshold.")
            if 'value' in position and position['value'] > self.thresholds.max_position_size:
                violations.append(f"Position {position['id']} exceeds position size threshold.")
        return violations