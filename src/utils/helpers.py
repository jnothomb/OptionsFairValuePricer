import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd

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

def calculate_historical_volatility(prices, window=30):
    """
    Calculate historical volatility from a series of prices
    
    Parameters:
    prices (pd.Series): Daily closing prices
    window (int): Rolling window in days
    
    Returns:
    pd.Series: Historical volatility (annualized)
    """
    returns = np.log(prices / prices.shift(1))
    vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return vol

def option_chain_analysis(options_data):
    """
    Analyze an option chain to find trading opportunities
    
    Parameters:
    options_data (pd.DataFrame): Option chain data with columns:
        - strike
        - bid
        - ask
        - volume
        - open_interest
        - implied_vol
    
    Returns:
    dict: Trading opportunities analysis
    """
    analysis = {
        'high_volume_strikes': [],
        'high_open_interest': [],
        'vol_skew': {},
        'potential_opportunities': []
    }
    
    # Find strikes with unusual volume
    volume_mean = options_data['volume'].mean()
    high_volume = options_data[options_data['volume'] > 2 * volume_mean]
    analysis['high_volume_strikes'] = high_volume['strike'].tolist()
    
    # Analyze volatility skew
    analysis['vol_skew'] = {
        'slope': np.polyfit(options_data['strike'], 
                          options_data['implied_vol'], 1)[0],
        'pattern': 'normal' if analysis['vol_skew']['slope'] < 0 else 'inverse'
    }
    
    return analysis

def risk_metrics(position_value, option_price, delta, gamma):
    """
    Calculate key risk metrics for an option position
    
    Parameters:
    position_value (float): Total position value
    option_price (float): Current option price
    delta (float): Option delta
    gamma (float): Option gamma
    
    Returns:
    dict: Risk metrics
    """
    return {
        'max_loss': min(position_value, option_price),
        'dollar_delta': position_value * delta,
        'dollar_gamma': position_value * gamma * 100,  # $gamma per 1% move
        'position_leverage': position_value / option_price
    }

def find_arbitrage_opportunities(calls, puts, spot, rf_rate):
    """
    Find potential arbitrage opportunities using put-call parity
    
    Parameters:
    calls (pd.DataFrame): Call options data
    puts (pd.DataFrame): Put options data
    spot (float): Current spot price
    rf_rate (float): Risk-free rate
    
    Returns:
    list: Potential arbitrage opportunities
    """
    opportunities = []
    
    for strike in calls['strike'].unique():
        call = calls[calls['strike'] == strike].iloc[0]
        put = puts[puts['strike'] == strike].iloc[0]
        
        # Put-call parity: C - P = S - K*e^(-rT)
        left_side = call['mid_price'] - put['mid_price']
        right_side = spot - strike * np.exp(-rf_rate * call['time_to_expiry'])
        
        if abs(left_side - right_side) > 0.05:  # 5 cents threshold
            opportunities.append({
                'strike': strike,
                'deviation': abs(left_side - right_side),
                'type': 'buy_call_sell_put' if left_side < right_side else 'buy_put_sell_call'
            })
    
    return opportunities

def calculate_breakeven(spot, strike, premium, option_type='call'):
    """
    Calculate breakeven price for an option position
    
    Parameters:
    spot (float): Current spot price
    strike (float): Option strike price
    premium (float): Option premium paid
    option_type (str): 'call' or 'put'
    
    Returns:
    float: Breakeven price
    """
    if option_type.lower() == 'call':
        return strike + premium
    else:
        return strike - premium