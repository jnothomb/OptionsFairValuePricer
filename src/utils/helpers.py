import numpy as np
from scipy.stats import norm

def implied_volatility(market_price, S, K, T, r, option_type='call', precision=0.0001, max_iter=100):
    """
    Calculate implied volatility using Newton-Raphson method
    
    Parameters:
    market_price (float): Observed market price of the option
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free rate
    option_type (str): 'call' or 'put'
    precision (float): Desired precision for implied volatility
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: Implied volatility
    """
    from src.models.black_scholes import BlackScholes
    
    sigma = 0.5  # Initial guess
    for i in range(max_iter):
        bs = BlackScholes(S, K, T, r, sigma)
        price = bs.call_price() if option_type == 'call' else bs.put_price()
        diff = market_price - price
        
        if abs(diff) < precision:
            return sigma
            
        # Vega calculation
        d1 = bs.d1
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        if vega == 0:
            return None
            
        sigma = sigma + diff/vega
        
        if sigma <= 0:
            sigma = 0.0001
            
    return None