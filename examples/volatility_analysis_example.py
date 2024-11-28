from src.utils.volatility import VolatilityAnalyzer
from src.utils.volatility_viz import VolatilityVisualizer
import pandas as pd
import numpy as np

def basic_volatility_analysis():
    """
    Basic example of volatility surface construction and analysis
    """
    # Create sample market data
    market_data = pd.DataFrame({
        'strike': [95, 100, 105] * 3,
        'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
        'option_type': ['call'] * 9,
        'underlying_price': [100] * 9
    })
    
    # Initialize analyzer
    analyzer = VolatilityAnalyzer(market_data)
    
    # Construct surface
    surface = analyzer.construct_vol_surface(method='cubic')
    
    # Visualize results
    viz = VolatilityVisualizer()
    viz.plot_volatility_surface(surface, title='Sample Volatility Surface')
    
    return surface

def advanced_volatility_analysis():
    """
    Advanced example including regime detection and forecasting
    """
    # Create historical data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    historical_data = pd.DataFrame({
        'date': dates,
        'price': np.random.lognormal(mean=4.6, sigma=0.2, size=len(dates)),
        'historical_vol': np.random.normal(0.2, 0.05, size=len(dates))
    })
    
    # Create market data with multiple strikes and expiries
    strikes = np.linspace(90, 110, 21)
    expiries = [0.25, 0.5, 1.0, 1.5, 2.0]
    
    market_data = []
    for strike in strikes:
        for expiry in expiries:
            # Simulate realistic implied vol surface
            atm_vol = 0.2
            skew = -0.02 * (strike - 100) / 100
            term = 0.01 * np.sqrt(expiry)
            implied_vol = atm_vol + skew + term
            
            market_data.append({
                'strike': strike,
                'expiry': expiry,
                'implied_vol': implied_vol,
                'option_type': 'call',
                'underlying_price': 100
            })
    
    market_data = pd.DataFrame(market_data)
    
    # Initialize analyzer with both market and historical data
    analyzer = VolatilityAnalyzer(market_data, historical_data)
    
    try:
        # Comprehensive analysis
        surface = analyzer.construct_vol_surface(method='svi')
        regime = analyzer.detect_volatility_regime(window=60)
        forecast = analyzer.forecast_volatility(horizon=30, method='ensemble')
        
        # Visualizations
        viz = VolatilityVisualizer()
        viz.plot_volatility_surface(surface)
        viz.plot_regime_analysis(
            historical_data['historical_vol'],
            pd.Series(index=dates, data='normal')  # Example regime
        )
        
        return {
            'surface': surface,
            'regime': regime,
            'forecast': forecast
        }
        
    except Exception as e:
        print(f"Error in volatility analysis: {str(e)}")
        return None

def error_handling_example():
    """
    Example demonstrating error handling in volatility analysis
    """
    try:
        # Attempt analysis with invalid data
        invalid_market_data = pd.DataFrame({
            'strike': [95, 100, 105],
            'expiry': [0.25, 0.25, 0.25]
            # Missing required columns
        })
        
        analyzer = VolatilityAnalyzer(invalid_market_data)
        
    except ValueError as e:
        print(f"Validation Error: {str(e)}")
    
    try:
        # Attempt surface construction with insufficient data
        sparse_market_data = pd.DataFrame({
            'strike': [100],
            'expiry': [0.25],
            'implied_vol': [0.2],
            'option_type': ['call'],
            'underlying_price': [100]
        })
        
        analyzer = VolatilityAnalyzer(sparse_market_data)
        surface = analyzer.construct_vol_surface()
        
    except ValueError as e:
        print(f"Surface Construction Error: {str(e)}")

if __name__ == "__main__":
    # Run basic analysis
    print("Running basic volatility analysis...")
    surface = basic_volatility_analysis()
    
    # Run advanced analysis
    print("\nRunning advanced volatility analysis...")
    results = advanced_volatility_analysis()
    
    # Demonstrate error handling
    print("\nDemonstrating error handling...")
    error_handling_example() 