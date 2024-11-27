import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes model parameters
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the stock (annual)
        
        Raises:
        ValueError: If any of the inputs are invalid
        """
        # Input validation
        if S <= 0:
            raise ValueError("Stock price must be positive")
        if K <= 0:
            raise ValueError("Strike price must be positive")
        if T <= 0:
            raise ValueError("Time to maturity must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
            
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        
        # Calculate d1 and d2
        self.d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)

    def call_price(self):
        """Calculate European call option price"""
        call = (self.S * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
        return call

    def put_price(self):
        """Calculate European put option price"""
        put = (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
               self.S * norm.cdf(-self.d1))
        return put

    def call_delta(self):
        """Calculate call option delta"""
        return norm.cdf(self.d1)

    def put_delta(self):
        """Calculate put option delta"""
        return -norm.cdf(-self.d1)
