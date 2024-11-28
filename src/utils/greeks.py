import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Greeks:
    """
    Container for option Greeks values
    
    Attributes:
        delta (float): Measures change in option price per $1 change in underlying
        gamma (float): Measures change in delta per $1 change in underlying
        theta (float): Measures change in option price per day (time decay)
        vega (float): Measures change in option price per 1% change in volatility
        rho (float): Measures change in option price per 1% change in interest rates
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class GreeksCalculator:
    """
    Calculator for option Greeks using Black-Scholes model
    
    The calculator uses the standard Black-Scholes partial derivatives
    to compute first-order Greeks plus Gamma. All calculations assume
    European-style options.
    """
    def __init__(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility):
        """
        Initialize Greeks calculator
        
        Parameters:
        spot_price (float): Current price of underlying
        strike (float): Strike price of option
        time_to_expiry (float): Time to expiration in years
        risk_free_rate (float): Risk-free interest rate
        volatility (float): Implied volatility
        """
        self.S = spot_price
        self.K = strike
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        
        # Calculate d1 and d2 (used in multiple Greeks calculations)
        self.d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)

    def calculate_greeks(self, option_type: str = 'call') -> Greeks:
        """
        Calculate all Greeks for an option position
        
        Args:
            option_type (str): 'call' or 'put'
            
        Returns:
            Greeks: Container with all calculated Greeks values
            
        Note: Signs are adjusted for puts vs calls automatically
        """
        is_call = option_type.lower() == 'call'
        
        delta = self._calculate_delta(is_call)
        gamma = self._calculate_gamma()
        theta = self._calculate_theta(is_call)
        vega = self._calculate_vega()
        rho = self._calculate_rho(is_call)
        
        return Greeks(delta, gamma, theta, vega, rho)

    def _calculate_delta(self, is_call: bool) -> float:
        """Calculate Delta"""
        from scipy.stats import norm
        sign = 1 if is_call else -1
        return sign * norm.cdf(sign * self.d1)

    def _calculate_gamma(self) -> float:
        """Calculate Gamma (same for calls and puts)"""
        from scipy.stats import norm
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def _calculate_theta(self, is_call: bool) -> float:
        """Calculate Theta"""
        from scipy.stats import norm
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        sign = 1 if is_call else -1
        term2 = sign * self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(sign * self.d2)
        return term1 - term2

    def _calculate_vega(self) -> float:
        """Calculate Vega (same for calls and puts)"""
        from scipy.stats import norm
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1)

    def _calculate_rho(self, is_call: bool) -> float:
        """Calculate Rho"""
        from scipy.stats import norm
        sign = 1 if is_call else -1
        return sign * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(sign * self.d2)

class PortfolioGreeks:
    def __init__(self, positions: List[Dict]):
        """
        Initialize portfolio Greeks calculator
        
        Parameters:
        positions (List[Dict]): List of position dictionaries containing:
            - option_type: 'call' or 'put'
            - quantity: number of contracts
            - spot_price: current price
            - strike: strike price
            - time_to_expiry: time to expiration in years
            - risk_free_rate: risk-free rate
            - volatility: implied volatility
        """
        self.positions = positions

    def calculate_portfolio_greeks(self) -> Greeks:
        """Calculate aggregate Greeks for the entire portfolio"""
        portfolio_greeks = Greeks(0, 0, 0, 0, 0)
        
        for position in self.positions:
            calculator = GreeksCalculator(
                position['spot_price'],
                position['strike'],
                position['time_to_expiry'],
                position['risk_free_rate'],
                position['volatility']
            )
            
            position_greeks = calculator.calculate_greeks(position['option_type'])
            quantity = position['quantity']
            
            # Aggregate Greeks
            portfolio_greeks.delta += position_greeks.delta * quantity
            portfolio_greeks.gamma += position_greeks.gamma * quantity
            portfolio_greeks.theta += position_greeks.theta * quantity
            portfolio_greeks.vega += position_greeks.vega * quantity
            portfolio_greeks.rho += position_greeks.rho * quantity
        
        return portfolio_greeks

    def risk_metrics(self) -> Dict:
        """Calculate risk metrics based on portfolio Greeks"""
        greeks = self.calculate_portfolio_greeks()
        
        return {
            'dollar_delta': greeks.delta * 100,  # Dollar change per 1 point move
            'gamma_risk': greeks.gamma * 100,    # Delta change per 1 point move
            'theta_day': greeks.theta / 365,     # Daily time decay
            'vega_risk': greeks.vega / 100,      # Price change per 1% vol change
            'rate_risk': greeks.rho / 100        # Price change per 1% rate change
        }
