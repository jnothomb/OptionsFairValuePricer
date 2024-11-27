import pytest
import numpy as np
from src.models.black_scholes import BlackScholes

def test_black_scholes_call():
    # Test case parameters
    S = 100    # Stock price
    K = 100    # Strike price
    T = 1      # One year to maturity
    r = 0.05   # 5% risk-free rate
    sigma = 0.2 # 20% volatility

    bs = BlackScholes(S, K, T, r, sigma)
    call_price = bs.call_price()
    
    # The call price should be positive and less than the stock price
    assert call_price > 0
    assert call_price < S

def test_put_call_parity():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    bs = BlackScholes(S, K, T, r, sigma)
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    # Test put-call parity: C - P = S - K*e^(-rT)
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)
    assert abs(left_side - right_side) < 1e-10

def test_delta_bounds():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    
    # Call delta should be between 0 and 1
    call_delta = bs.call_delta()
    assert 0 <= call_delta <= 1
    
    # Put delta should be between -1 and 0
    put_delta = bs.put_delta()
    assert -1 <= put_delta <= 0

def test_invalid_inputs():
    with pytest.raises(ValueError):
        BlackScholes(S=-100, K=100, T=1, r=0.05, sigma=0.2)
    
    with pytest.raises(ValueError):
        BlackScholes(S=100, K=100, T=-1, r=0.05, sigma=0.2)
        
    with pytest.raises(ValueError):
        BlackScholes(S=100, K=100, T=1, r=0.05, sigma=-0.2)