from dataclasses import dataclass
from typing import Dict, List, Any
import logging
from src.utils.greeks import Greeks
import numpy as np

@dataclass
class RiskThresholds:
    """
    Define risk management thresholds for option positions.

    Attributes:
        max_position_delta (float): Maximum absolute delta exposure.
        max_position_gamma (float): Maximum absolute gamma exposure.
        max_position_vega (float): Maximum absolute vega exposure.
        max_position_theta (float): Maximum negative theta exposure.
        max_position_size (float): Maximum position value.
        max_loss_threshold (float): Maximum allowable loss before action is required.
    """
    max_position_delta: float = 1.0    # Default maximum delta exposure
    max_position_gamma: float = 0.1     # Default maximum gamma exposure
    max_position_vega: float = 0.5      # Default maximum vega exposure
    max_position_theta: float = -0.2     # Default maximum negative theta exposure
    max_position_size: float = 100000    # Default maximum position size
    max_loss_threshold: float = 5000     # Default maximum loss threshold

    def __post_init__(self):
        # Validate thresholds to ensure they are sensible
        if self.max_position_delta < 0:
            raise ValueError("max_position_delta must be non-negative.")
        if self.max_position_gamma < 0:
            raise ValueError("max_position_gamma must be non-negative.")
        if self.max_position_vega < 0:
            raise ValueError("max_position_vega must be non-negative.")
        if self.max_position_theta >= 0:
            raise ValueError("max_position_theta must be negative.")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive.")
        if self.max_loss_threshold < 0:
            raise ValueError("max_loss_threshold must be non-negative.")

class RiskManager:
    """
    Risk management system for options positions
    
    Provides:
    - Risk threshold monitoring
    - Position adjustment suggestions
    - Risk violation alerts
    - Position size management
    """
    def __init__(self, thresholds: RiskThresholds):
        self.thresholds = thresholds
        self.positions = []  # Initialize an empty list to hold positions
        self.risk_metrics = {}  # Initialize a dictionary to hold risk metrics
        self.logger = logging.getLogger(__name__)
        
    def check_position_risks(self, greeks: 'Greeks', 
                           position_value: float) -> Dict[str, bool]:
        """Check if position Greeks exceed thresholds"""
        violations = {
            'delta': abs(greeks.delta) > self.thresholds.max_position_delta,
            'gamma': abs(greeks.gamma) > self.thresholds.max_position_gamma,
            'theta': greeks.theta < self.thresholds.max_position_theta,
            'vega': abs(greeks.vega) > self.thresholds.max_position_vega,
            'position_size': position_value > self.thresholds.max_position_size
        }
        
        for greek, is_violated in violations.items():
            if is_violated:
                self.logger.warning(f"Risk threshold violated: {greek}")
        
        return violations

    def suggest_position_adjustments(self, greeks: 'Greeks', 
                                   position_value: float) -> List[str]:
        """Suggest adjustments based on risk violations"""
        violations = self.check_position_risks(greeks, position_value)
        suggestions = []
        
        if violations['delta']:
            suggestions.append("Consider delta hedge")
        if violations['gamma']:
            suggestions.append("Reduce gamma exposure by widening strikes")
        if violations['theta']:
            suggestions.append("Reduce negative theta by closing short options")
        if violations['vega']:
            suggestions.append("Reduce vega exposure by adding opposing positions")
        
        return suggestions

    def calculate_position_size(self, account_balance, risk_per_trade, entry_price, stop_loss_price):
        """
        Calculate the position size based on account balance, risk per trade, entry price, and stop loss price.

        Args:
            account_balance (float): Total account balance.
            risk_per_trade (float): Amount of capital to risk on a single trade.
            entry_price (float): Entry price of the position.
            stop_loss_price (float): Stop loss price of the position.

        Returns:
            float: Calculated position size.
        """
        risk_per_share = entry_price - stop_loss_price
        position_size = risk_per_trade / risk_per_share
        return position_size

    def calculate_margin_requirement(self, position_value, margin_rate):
        """
        Calculate the margin requirement for a given position.

        Args:
            position_value (float): Total value of the position.
            margin_rate (float): Margin rate (as a decimal).

        Returns:
            float: Required margin for the position.
        """
        return position_value * margin_rate

    def calculate_var(self, portfolio_returns, confidence_level=0.95):
        """Calculate Value at Risk (VaR) for a portfolio."""
        if not isinstance(portfolio_returns, np.ndarray):
            portfolio_returns = np.array(portfolio_returns)
        
        # Calculate the VaR at the specified confidence level
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def run_stress_tests(self, scenarios):
        """Run stress tests on the portfolio based on predefined scenarios."""
        results = {}
        for scenario, changes in scenarios.items():
            # Simulate the impact of the scenario on the portfolio
            # This is a simplified example; actual implementation may vary
            simulated_portfolio_value = self.portfolio_value * (1 + changes['price_change'])
            results[scenario] = {
                'simulated_value': simulated_portfolio_value,
                'loss': self.portfolio_value - simulated_portfolio_value,
                'pnl_impact': changes['price_change'] * self.portfolio_value  # Example PnL impact calculation
            }
        return results

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

    def remove_position(self, position_id: int):
        """Remove a position by its ID."""
        self.positions = [pos for pos in self.positions if pos['id'] != position_id]

    def calculate_position_risk(self, position_id: int) -> Dict[str, float]:
        """Calculate risk metrics for a specific position."""
        position = next((pos for pos in self.positions if pos['id'] == position_id), None)
        if position is None:
            raise ValueError("Position not found.")
        
        # Placeholder for actual risk calculation logic
        return {
            'delta': position.get('delta', 0.0),  # Replace with actual delta calculation
            'gamma': position.get('gamma', 0.0),  # Replace with actual gamma calculation
            'theta': position.get('theta', 0.0),  # Replace with actual theta calculation
            'vega': position.get('vega', 0.0)     # Replace with actual vega calculation
        }

    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate risk metrics for the entire portfolio."""
        # Placeholder for actual portfolio risk calculation logic
        # You would replace this with actual calculations based on all positions
        return {
            'total_delta': 0.0,  # Replace with actual total delta calculation
            'total_gamma': 0.0,  # Replace with actual total gamma calculation
            'total_theta': 0.0,  # Replace with actual total theta calculation
            'total_vega': 0.0,   # Replace with actual total vega calculation
            'total_exposure': 0.0 # Replace with actual total exposure calculation
        }

    def check_risk_violations(self) -> List[str]:
        """Check for risk threshold violations."""
        violations = []
        for position in self.positions:
            # Example checks (you will need to implement actual logic based on your thresholds)
            if abs(position['delta']) > self.thresholds.max_position_delta:
                violations.append(f"Position {position['id']} exceeds delta threshold.")
            if abs(position['gamma']) > self.thresholds.max_position_gamma:
                violations.append(f"Position {position['id']} exceeds gamma threshold.")
            if position['theta'] < self.thresholds.max_position_theta:
                violations.append(f"Position {position['id']} exceeds theta threshold.")
            if abs(position['vega']) > self.thresholds.max_position_vega:
                violations.append(f"Position {position['id']} exceeds vega threshold.")
            if position['value'] > self.thresholds.max_position_size:
                violations.append(f"Position {position['id']} exceeds position size threshold.")
        return violations

    def suggest_risk_adjustments(self) -> Dict[str, Any]:
        """Suggest adjustments to mitigate risk."""
        return {
            'suggested_trades': [],  # Replace with actual suggestions
            'risk_impact': 0.0       # Replace with actual risk impact calculation
        }

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate a risk report for the current positions."""
        report = {
            'portfolio_risk': self.calculate_portfolio_risk(),
            'risk_violations': self.check_risk_violations(),
            'stress_test_results': self.run_stress_tests({}),
            'suggested_actions': self.suggest_risk_adjustments()
        }
        return report