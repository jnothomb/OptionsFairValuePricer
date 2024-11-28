from src.models.black_scholes import BlackScholes
from examples.trading_strategies import OptionStrategies

def analyze_iron_condor():
    """
    Example analysis of an iron condor strategy
    
    Demonstrates:
    - Strategy creation
    - Greeks calculation
    - Risk analysis
    - Visualization generation
    - Position monitoring
    
    This example can be used as a template for analyzing other strategies
    
    Usage:
        python strategy_analysis_example.py
    """
    # Initialize strategy
    spot_price = 100
    risk_free_rate = 0.05
    volatility = 0.2
    days_to_expiry = 30
    
    strategies = OptionStrategies(spot_price, risk_free_rate)
    
    # Create and analyze iron condor
    ic_trade = strategies.iron_condor(
        volatility=volatility,
        days_to_expiry=days_to_expiry
    )
    
    # Print analysis
    print("\nStrategy Analysis:")
    print("-----------------")
    print(f"Greeks:")
    for greek, value in ic_trade['analysis']['greeks'].__dict__.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    print("\nRisk Metrics:")
    print("------------")
    for metric, value in ic_trade['analysis']['risk_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nRisk Violations:")
    print("--------------")
    for greek, violated in ic_trade['analysis']['risk_violations'].items():
        status = "VIOLATED" if violated else "OK"
        print(f"{greek.capitalize()}: {status}")
    
    print("\nSuggested Adjustments:")
    print("--------------------")
    for adjustment in ic_trade['analysis']['suggested_adjustments']:
        print(f"- {adjustment}")
    
    # Show visualizations
    ic_trade['analysis']['visualizations']['greeks_sensitivity'].show()

if __name__ == "__main__":
    analyze_iron_condor() 