from dataclasses import dataclass
from typing import Dict, List
import logging
from src.utils.greeks import Greeks

@dataclass
class RiskThresholds:
    """
    Define risk management thresholds for option positions
    
    Default values are conservative and should be adjusted based on:
    - Account size
    - Risk tolerance
    - Market conditions
    - Strategy objectives
    """
    max_delta: float = 100    # Maximum absolute delta exposure
    max_gamma: float = 10     # Maximum absolute gamma exposure
    max_theta: float = -50    # Maximum negative theta exposure
    max_vega: float = 100     # Maximum absolute vega exposure
    max_position_size: float = 10000  # Maximum position value

class RiskManager:
    """
    Risk management system for options positions
    
    Provides:
    - Risk threshold monitoring
    - Position adjustment suggestions
    - Risk violation alerts
    - Position size management
    """
    def __init__(self, thresholds: RiskThresholds = None):
        self.thresholds = thresholds or RiskThresholds()
        self.logger = logging.getLogger(__name__)
        
    def check_position_risks(self, greeks: 'Greeks', 
                           position_value: float) -> Dict[str, bool]:
        """Check if position Greeks exceed thresholds"""
        violations = {
            'delta': abs(greeks.delta) > self.thresholds.max_delta,
            'gamma': abs(greeks.gamma) > self.thresholds.max_gamma,
            'theta': greeks.theta < self.thresholds.max_theta,
            'vega': abs(greeks.vega) > self.thresholds.max_vega,
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