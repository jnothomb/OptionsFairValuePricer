import pandas as pd
import numpy as np
from src.models.black_scholes import BlackScholes
from src.utils.helpers import (
    calculate_historical_volatility,
    option_chain_analysis,
    risk_metrics,
    calculate_breakeven
)
from src.utils.probability import ProbabilityCalculator
from src.utils.greeks import GreeksCalculator, PortfolioGreeks
from src.utils.visualizations import GreeksVisualizer
from src.utils.risk_manager import RiskManager, RiskThresholds
from typing import List, Dict

"""
1. An OptionStrategies class with common option strategies:
    - Iron Condor (neutral strategy)
    - Bull Call Spread (directional strategy)
    - Volatility-based strategy selection
2. Each strategy includes:
    - Risk/reward calculations
    - Breakeven points
    - Maximum profit/loss
    - Strategy recommendations based on volatility analysis
3. Practical considerations:
    - Uses both historical and implied volatility
    - Includes position sizing and risk metrics
    - Provides clear breakeven points and profit zones
"""

class OptionStrategies:
    """
    Implementation of various option trading strategies with risk analysis
    
    Features:
    - Strategy construction
    - Greeks calculation
    - Risk analysis
    - Position visualization
    - Adjustment suggestions
    
    Each strategy includes:
    - Probability analysis
    - Greeks analysis
    - Risk metrics
    - Visual analytics
    """
    
    def __init__(self, spot_price, risk_free_rate):
        self.spot = spot_price
        self.rf_rate = risk_free_rate
        self.risk_manager = RiskManager()
        self.visualizer = GreeksVisualizer()

    def _calculate_strategy_greeks(self, strategy_positions):
        """Calculate Greeks for a strategy"""
        portfolio = PortfolioGreeks(strategy_positions)
        greeks = portfolio.calculate_portfolio_greeks()
        risk_metrics = portfolio.risk_metrics()
        
        return {
            'greeks': greeks,
            'risk_metrics': risk_metrics
        }

    def analyze_strategy(self, strategy_name: str, positions: List[Dict], 
                        strategy_value: float):
        """
        Comprehensive strategy analysis including Greeks and risk metrics
        
        Args:
            strategy_name: Name of the strategy being analyzed
            positions: List of option positions in the strategy
            strategy_value: Total value of the strategy
            
        Returns:
            Dict containing:
            - Greeks values
            - Risk metrics
            - Risk violations
            - Suggested adjustments
            - Visualization objects
            
        Note: This is the main analysis entry point for all strategies
        """
        # Calculate Greeks
        portfolio = PortfolioGreeks(positions)
        greeks = portfolio.calculate_portfolio_greeks()
        risk_metrics = portfolio.risk_metrics()
        
        # Risk analysis
        risk_violations = self.risk_manager.check_position_risks(
            greeks, strategy_value
        )
        adjustments = self.risk_manager.suggest_position_adjustments(
            greeks, strategy_value
        )
        
        # Create visualizations
        price_range = np.linspace(self.spot * 0.8, self.spot * 1.2, 100)
        greeks_plot = self.visualizer.plot_greeks_sensitivity(
            positions, price_range
        )
        
        return {
            'strategy_name': strategy_name,
            'greeks': greeks,
            'risk_metrics': risk_metrics,
            'risk_violations': risk_violations,
            'suggested_adjustments': adjustments,
            'visualizations': {
                'greeks_sensitivity': greeks_plot
            }
        }

    def iron_condor(self, volatility, days_to_expiry, width=5, wing_distance=10):
        """
        Create an iron condor strategy with probability analysis
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        width (int): Width between short and long strikes
        wing_distance (int): Distance of short strikes from current price
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        put_short = self.spot - wing_distance
        put_long = put_short - width
        call_short = self.spot + wing_distance
        call_long = call_short + width
        
        # Calculate option prices
        bs_put_short = BlackScholes(self.spot, put_short, T, self.rf_rate, volatility)
        bs_put_long = BlackScholes(self.spot, put_long, T, self.rf_rate, volatility)
        bs_call_short = BlackScholes(self.spot, call_short, T, self.rf_rate, volatility)
        bs_call_long = BlackScholes(self.spot, call_long, T, self.rf_rate, volatility)
        
        # Calculate premium collected
        premium = (bs_put_short.put_price() - bs_put_long.put_price() +
                  bs_call_short.call_price() - bs_call_long.call_price())
        
        # Calculate max risk (width of either spread minus premium)
        max_risk = width - premium
        
        # Add probability calculations
        prob_calc = ProbabilityCalculator(
            spot_price=self.spot,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            risk_free_rate=self.rf_rate
        )
        
        # Calculate probability of profit
        pop = prob_calc.prob_between_prices(put_short, call_short)
        
        # Calculate expected value
        def payoff_func(price):
            if price <= put_long:
                return -width
            elif price <= put_short:
                return price - put_long - width
            elif price <= call_short:
                return premium
            elif price <= call_long:
                return call_long - price + premium
            else:
                return -width
        
        expected_value = prob_calc.expected_value(payoff_func)
        
        # Add positions for Greeks calculation
        positions = [
            {
                'option_type': 'put',
                'quantity': -1,
                'spot_price': self.spot,
                'strike': put_short,
                'time_to_expiry': days_to_expiry/365,
                'risk_free_rate': self.rf_rate,
                'volatility': volatility
            },
            {
                'option_type': 'put',
                'quantity': 1,
                'spot_price': self.spot,
                'strike': put_long,
                'time_to_expiry': days_to_expiry/365,
                'risk_free_rate': self.rf_rate,
                'volatility': volatility
            },
            # Add call positions similarly
        ]
        
        greeks_analysis = self._calculate_strategy_greeks(positions)
        
        # Add probability metrics to the return dictionary
        result = {
            'strategy': 'Iron Condor',
            'premium_collected': premium,
            'max_risk': max_risk,
            'max_return': premium,
            'breakeven_lower': put_short - premium,
            'breakeven_upper': call_short + premium,
            'profit_zone': f"Between {put_short} and {call_short}",
            'risk_reward_ratio': max_risk / premium,
            'probability_of_profit': pop,
            'expected_value': expected_value,
            'probability_metrics': {
                'prob_below_lower_break': prob_calc.prob_below_price(put_short - premium),
                'prob_above_upper_break': prob_calc.prob_above_price(call_short + premium),
                'prob_in_profit_zone': pop,
                'optimal_prob': prob_calc.prob_between_prices(
                    put_short + premium/2,
                    call_short - premium/2
                )
            },
            'greeks_analysis': greeks_analysis
        }
        
        analysis = self.analyze_strategy(
            'Iron Condor',
            positions,
            position_value=max_risk
        )
        
        return {
            **result,  # Previous strategy results
            'analysis': analysis
        }

    def bull_call_spread(self, volatility, days_to_expiry, width=5):
        """
        Create a bull call spread strategy
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        width (int): Width between long and short strikes
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        long_strike = self.spot
        short_strike = long_strike + width
        
        # Calculate option prices
        bs_long = BlackScholes(self.spot, long_strike, T, self.rf_rate, volatility)
        bs_short = BlackScholes(self.spot, short_strike, T, self.rf_rate, volatility)
        
        # Calculate net debit
        net_debit = bs_long.call_price() - bs_short.call_price()
        
        return {
            'strategy': 'Bull Call Spread',
            'net_debit': net_debit,
            'max_risk': net_debit,
            'max_return': width - net_debit,
            'breakeven': long_strike + net_debit,
            'risk_reward_ratio': net_debit / (width - net_debit)
        }

    def volatility_strategy(self, historical_prices, current_implied_vol, 
                          days_to_expiry, threshold=0.1):
        """
        Identify volatility-based trading opportunities
        
        Parameters:
        historical_prices (pd.Series): Historical price data
        current_implied_vol (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        threshold (float): Threshold for vol difference
        
        Returns:
        dict: Strategy recommendation
        """
        hist_vol = calculate_historical_volatility(historical_prices).iloc[-1]
        vol_difference = current_implied_vol - hist_vol
        
        strategy = {
            'historical_volatility': hist_vol,
            'implied_volatility': current_implied_vol,
            'vol_difference': vol_difference
        }
        
        if vol_difference > threshold:
            strategy['recommendation'] = 'Sell Volatility - Consider Iron Condor'
            strategy['suggested_strategy'] = self.iron_condor(
                current_implied_vol, days_to_expiry
            )
        elif vol_difference < -threshold:
            strategy['recommendation'] = 'Buy Volatility - Consider Straddle/Strangle'
            # Add straddle implementation here
        else:
            strategy['recommendation'] = 'Neutral - No Clear Volatility Edge'
            
        return strategy

    def butterfly_spread(self, volatility, days_to_expiry, wing_width=5):
        """
        Create a butterfly spread (long 1 ITM call + long 1 OTM call + short 2 ATM calls)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        wing_width (int): Distance between strikes
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        lower_strike = self.spot - wing_width
        middle_strike = self.spot
        upper_strike = self.spot + wing_width
        
        # Calculate option prices
        bs_lower = BlackScholes(self.spot, lower_strike, T, self.rf_rate, volatility)
        bs_middle = BlackScholes(self.spot, middle_strike, T, self.rf_rate, volatility)
        bs_upper = BlackScholes(self.spot, upper_strike, T, self.rf_rate, volatility)
        
        # Calculate net debit
        net_debit = (bs_lower.call_price() + bs_upper.call_price() - 
                    2 * bs_middle.call_price())
        
        return {
            'strategy': 'Butterfly Spread',
            'net_debit': net_debit,
            'max_risk': net_debit,
            'max_return': wing_width - net_debit,
            'breakeven_lower': lower_strike + net_debit,
            'breakeven_upper': upper_strike - net_debit,
            'optimal_price': middle_strike,
            'profit_zone': f"Between {lower_strike + net_debit} and {upper_strike - net_debit}"
        }

    def calendar_spread(self, volatility, near_term_days, far_term_days):
        """
        Create a calendar spread (short near-term, long far-term)
        
        Parameters:
        volatility (float): Current implied volatility
        near_term_days (int): Days until near-term expiration
        far_term_days (int): Days until far-term expiration
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T_near = near_term_days / 365
        T_far = far_term_days / 365
        strike = self.spot  # ATM strike
        
        # Calculate option prices
        bs_near = BlackScholes(self.spot, strike, T_near, self.rf_rate, volatility)
        bs_far = BlackScholes(self.spot, strike, T_far, self.rf_rate, volatility)
        
        # Calculate net debit
        net_debit = bs_far.call_price() - bs_near.call_price()
        
        return {
            'strategy': 'Calendar Spread',
            'net_debit': net_debit,
            'max_risk': net_debit,
            'near_term_expiry': near_term_days,
            'far_term_expiry': far_term_days,
            'strike': strike,
            'optimal_scenario': 'Stock near strike at near-term expiration'
        }

    def strangle(self, volatility, days_to_expiry, strike_distance=10):
        """
        Create a long strangle (OTM call + OTM put)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        strike_distance (int): Distance of strikes from current price
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        put_strike = self.spot - strike_distance
        call_strike = self.spot + strike_distance
        
        bs_put = BlackScholes(self.spot, put_strike, T, self.rf_rate, volatility)
        bs_call = BlackScholes(self.spot, call_strike, T, self.rf_rate, volatility)
        
        total_cost = bs_put.put_price() + bs_call.call_price()
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price <= put_strike:
                return put_strike - price - total_cost
            elif price <= call_strike:
                return -total_cost
            else:
                return price - call_strike - total_cost
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=put_strike - total_cost,
            breakeven_upper=call_strike + total_cost
        )
        
        return {
            'strategy': 'Long Strangle',
            'total_cost': total_cost,
            'max_risk': total_cost,
            'max_return': 'Unlimited',
            'breakeven_lower': put_strike - total_cost,
            'breakeven_upper': call_strike + total_cost,
            'profit_zones': f"Below {put_strike - total_cost} or above {call_strike + total_cost}",
            'best_scenario': 'Large move in either direction',
            'implied_vol': volatility,
            'delta': bs_put.put_delta() + bs_call.call_delta(),  # Should be near zero
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def jade_lizard(self, volatility, days_to_expiry, call_spread_width=5):
        """
        Create a jade lizard (short put + bull call spread)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        call_spread_width (int): Width of call spread
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        put_strike = self.spot - 5  # Slightly OTM put
        call_strike1 = self.spot + 5  # First call strike
        call_strike2 = call_strike1 + call_spread_width  # Second call strike
        
        bs_put = BlackScholes(self.spot, put_strike, T, self.rf_rate, volatility)
        bs_call1 = BlackScholes(self.spot, call_strike1, T, self.rf_rate, volatility)
        bs_call2 = BlackScholes(self.spot, call_strike2, T, self.rf_rate, volatility)
        
        credit = (bs_put.put_price() + bs_call1.call_price() - 
                 bs_call2.call_price())
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price <= put_strike:
                return put_strike - price - credit
            elif price <= call_strike1:
                return credit
            elif price <= call_strike2:
                return credit - (price - call_strike1)
            else:
                return credit - call_spread_width
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=put_strike - credit,
            breakeven_upper=call_strike1 + credit,
            max_profit_lower=put_strike,
            max_profit_upper=call_strike1
        )
        
        return {
            'strategy': 'Jade Lizard',
            'net_credit': credit,
            'max_return': credit,
            'max_risk': put_strike - credit,
            'breakeven_lower': put_strike - credit,
            'breakeven_upper': call_strike1 + credit,
            'profit_zone': f"Between {put_strike - credit} and {call_strike1 + credit}",
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def call_backspread(self, volatility, days_to_expiry, ratio=2, strike_distance=5):
        """
        Create a call ratio backspread (short 1 ITM call, long 2+ OTM calls)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        ratio (int): Number of long calls per short call
        strike_distance (int): Distance between strikes
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        short_strike = self.spot - strike_distance  # ITM
        long_strike = self.spot + strike_distance   # OTM
        
        bs_short = BlackScholes(self.spot, short_strike, T, self.rf_rate, volatility)
        bs_long = BlackScholes(self.spot, long_strike, T, self.rf_rate, volatility)
        
        # Calculate net debit/credit
        net_cost = ratio * bs_long.call_price() - bs_short.call_price()
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price <= short_strike:
                return -net_cost
            elif price <= long_strike:
                return (price - short_strike) - net_cost
            else:
                return (ratio * (price - long_strike) - (price - short_strike)) - net_cost
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=short_strike + net_cost,
            breakeven_upper=long_strike + (net_cost / (ratio - 1)),
            max_profit_lower=long_strike + (net_cost / (ratio - 1))  # Unlimited upside
        )
        
        return {
            'strategy': 'Call Ratio Backspread',
            'ratio': f"1:{ratio}",
            'net_cost': net_cost,
            'max_risk': (long_strike - short_strike) - net_cost if net_cost > 0 else -net_cost,
            'max_return': 'Unlimited',
            'breakeven_points': [
                short_strike + net_cost,
                long_strike + (net_cost / (ratio - 1))
            ],
            'profit_zones': f"Above {long_strike + (net_cost / (ratio - 1))}",
            'best_scenario': 'Strong upward move',
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def put_backspread(self, volatility, days_to_expiry, ratio=2, strike_distance=5):
        """
        Create a put ratio backspread (short 1 OTM put, long 2+ ITM puts)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        ratio (int): Number of long puts per short put
        strike_distance (int): Distance between strikes
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        short_strike = self.spot + strike_distance  # OTM
        long_strike = self.spot - strike_distance   # ITM
        
        bs_short = BlackScholes(self.spot, short_strike, T, self.rf_rate, volatility)
        bs_long = BlackScholes(self.spot, long_strike, T, self.rf_rate, volatility)
        
        net_cost = ratio * bs_long.put_price() - bs_short.put_price()
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price >= short_strike:
                return -net_cost
            elif price >= long_strike:
                return (short_strike - price) - net_cost
            else:
                return (ratio * (long_strike - price) - (short_strike - price)) - net_cost
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=long_strike - (net_cost / (ratio - 1)),
            breakeven_upper=short_strike - net_cost,
            max_profit_upper=long_strike - (net_cost / (ratio - 1))  # Maximum profit on downside
        )
        
        return {
            'strategy': 'Put Ratio Backspread',
            'ratio': f"1:{ratio}",
            'net_cost': net_cost,
            'max_risk': (short_strike - long_strike) - net_cost if net_cost > 0 else -net_cost,
            'max_return': long_strike * (ratio - 1) - net_cost,
            'breakeven_points': [
                long_strike - (net_cost / (ratio - 1)),
                short_strike - net_cost
            ],
            'profit_zones': f"Below {long_strike - (net_cost / (ratio - 1))}",
            'best_scenario': 'Strong downward move',
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def broken_wing_butterfly(self, volatility, days_to_expiry, wing_width=5, broken_width=7):
        """
        Create a broken wing butterfly (asymmetric butterfly spread)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        wing_width (int): Standard wing width
        broken_width (int): Width of the broken wing
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        
        # Define strikes
        lower_strike = self.spot - wing_width
        middle_strike = self.spot
        upper_strike = self.spot + broken_width
        
        bs_lower = BlackScholes(self.spot, lower_strike, T, self.rf_rate, volatility)
        bs_middle = BlackScholes(self.spot, middle_strike, T, self.rf_rate, volatility)
        bs_upper = BlackScholes(self.spot, upper_strike, T, self.rf_rate, volatility)
        
        net_debit = (bs_lower.call_price() + bs_upper.call_price() - 
                    2 * bs_middle.call_price())
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price <= lower_strike:
                return -net_debit
            elif price <= middle_strike:
                return (price - lower_strike) - net_debit
            elif price <= upper_strike:
                return (upper_strike - price) - net_debit
            else:
                return -net_debit
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=lower_strike + net_debit,
            breakeven_upper=upper_strike - net_debit,
            max_profit_lower=middle_strike,
            max_profit_upper=middle_strike
        )
        
        return {
            'strategy': 'Broken Wing Butterfly',
            'net_debit': net_debit,
            'max_risk': max(net_debit, broken_width - wing_width - net_debit),
            'max_return': wing_width - net_debit,
            'breakeven_lower': lower_strike + net_debit,
            'breakeven_upper': upper_strike - net_debit,
            'optimal_price': middle_strike,
            'profit_zone': f"Between {lower_strike + net_debit} and {upper_strike - net_debit}",
            'skew': 'Upside' if broken_width > wing_width else 'Downside',
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def long_straddle(self, volatility, days_to_expiry):
        """
        Create a long straddle (long ATM call + long ATM put)
        
        Parameters:
        volatility (float): Current implied volatility
        days_to_expiry (int): Days until expiration
        
        Returns:
        dict: Strategy details and risk metrics
        """
        T = days_to_expiry / 365
        strike = self.spot  # ATM strike
        
        bs = BlackScholes(self.spot, strike, T, self.rf_rate, volatility)
        
        call_price = bs.call_price()
        put_price = bs.put_price()
        total_cost = call_price + put_price
        
        prob_calc = self._get_probability_calculator(volatility, days_to_expiry)
        
        def payoff_function(price):
            if price <= strike:
                return max(strike - price, 0) - total_cost
            else:
                return max(price - strike, 0) - total_cost
        
        prob_metrics = prob_calc.strategy_metrics(
            breakeven_lower=strike - total_cost,
            breakeven_upper=strike + total_cost
        )
        
        # Add additional probability metrics specific to straddle
        prob_metrics.update({
            'prob_move_greater_than_cost': (
                prob_calc.prob_below_price(strike - total_cost) +
                prob_calc.prob_above_price(strike + total_cost)
            ),
            'prob_one_std_move': (
                1 - prob_calc.prob_between_prices(
                    strike * (1 - volatility * np.sqrt(days_to_expiry/365)),
                    strike * (1 + volatility * np.sqrt(days_to_expiry/365))
                )
            )
        })
        
        return {
            'strategy': 'Long Straddle',
            'total_cost': total_cost,
            'max_risk': total_cost,
            'max_return': 'Unlimited',
            'breakeven_lower': strike - total_cost,
            'breakeven_upper': strike + total_cost,
            'profit_zones': f"Below {strike - total_cost} or above {strike + total_cost}",
            'best_scenario': 'Large move in either direction',
            'implied_vol': volatility,
            'delta': bs.call_delta() + bs.put_delta(),  # Should be near zero
            'probability_metrics': prob_metrics,
            'expected_value': prob_calc.expected_value(payoff_function)
        }

    def _get_probability_calculator(self, volatility, days_to_expiry):
        return ProbabilityCalculator(
            spot_price=self.spot,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            risk_free_rate=self.rf_rate
        )

def example_usage():
    # Example usage of the strategies
    spot_price = 100
    rf_rate = 0.05
    volatility = 0.2
    days_to_expiry = 30
    
    strategies = OptionStrategies(spot_price, rf_rate)
    
    # Create an iron condor
    iron_condor = strategies.iron_condor(volatility, days_to_expiry)
    print("\nIron Condor Strategy:")
    for key, value in iron_condor.items():
        print(f"{key}: {value}")
    
    # Create a bull call spread
    bull_spread = strategies.bull_call_spread(volatility, days_to_expiry)
    print("\nBull Call Spread Strategy:")
    for key, value in bull_spread.items():
        print(f"{key}: {value}")
    
    # Test new strategies
    butterfly = strategies.butterfly_spread(volatility, days_to_expiry)
    print("\nButterfly Spread Strategy:")
    for key, value in butterfly.items():
        print(f"{key}: {value}")
    
    calendar = strategies.calendar_spread(volatility, 30, 60)
    print("\nCalendar Spread Strategy:")
    for key, value in calendar.items():
        print(f"{key}: {value}")
    
    strangle = strategies.strangle(volatility, days_to_expiry)
    print("\nStrangle Strategy:")
    for key, value in strangle.items():
        print(f"{key}: {value}")
    
    jade_lizard = strategies.jade_lizard(volatility, days_to_expiry)
    print("\nJade Lizard Strategy:")
    for key, value in jade_lizard.items():
        print(f"{key}: {value}")
    
    call_backspread = strategies.call_backspread(volatility, days_to_expiry)
    print("\nCall Backspread Strategy:")
    for key, value in call_backspread.items():
        print(f"{key}: {value}")
    
    put_backspread = strategies.put_backspread(volatility, days_to_expiry)
    print("\nPut Backspread Strategy:")
    for key, value in put_backspread.items():
        print(f"{key}: {value}")
    
    broken_wing = strategies.broken_wing_butterfly(volatility, days_to_expiry)
    print("\nBroken Wing Butterfly Strategy:")
    for key, value in broken_wing.items():
        print(f"{key}: {value}")
    
    straddle = strategies.long_straddle(volatility, days_to_expiry)
    print("\nLong Straddle Strategy:")
    for key, value in straddle.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()

"""
Use this by running:

from examples.trading_strategies import OptionStrategies

# Initialize with current market conditions
strategies = OptionStrategies(
    spot_price=100,
    risk_free_rate=0.05
)

# Get iron condor setup
ic_trade = strategies.iron_condor(
    volatility=0.2,
    days_to_expiry=30,
    width=5,
    wing_distance=10
)

print(ic_trade)
"""