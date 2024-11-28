import numpy as np
from scipy.stats import norm

class ProbabilityCalculator:
    def __init__(self, spot_price, volatility, days_to_expiry, risk_free_rate):
        self.S = spot_price
        self.sigma = volatility
        self.T = days_to_expiry / 365
        self.r = risk_free_rate

    def prob_above_price(self, price):
        """Calculate probability of being above a certain price at expiration"""
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        vol_sqrt_t = self.sigma * np.sqrt(self.T)
        
        z_score = (np.log(price/self.S) - drift) / vol_sqrt_t
        return 1 - norm.cdf(z_score)

    def prob_below_price(self, price):
        """Calculate probability of being below a certain price at expiration"""
        return 1 - self.prob_above_price(price)

    def prob_between_prices(self, lower_price, upper_price):
        """Calculate probability of being between two prices at expiration"""
        return self.prob_above_price(lower_price) - self.prob_above_price(upper_price)

    def expected_value(self, strategy_payoff_func):
        """Calculate expected value using numerical integration"""
        std_dev = self.sigma * np.sqrt(self.T)
        price_range = np.linspace(self.S * 0.5, self.S * 1.5, 1000)
        
        total_probability = 0
        total_expected_value = 0
        
        for price in price_range:
            prob_density = norm.pdf(
                np.log(price/self.S), 
                (self.r - 0.5 * self.sigma**2) * self.T, 
                std_dev
            )
            payoff = strategy_payoff_func(price)
            total_expected_value += payoff * prob_density
            total_probability += prob_density
            
        return total_expected_value / total_probability

    def strategy_metrics(self, breakeven_lower, breakeven_upper, max_profit_lower=None, max_profit_upper=None):
        """Calculate comprehensive strategy probability metrics"""
        if max_profit_lower is None:
            max_profit_lower = breakeven_lower
        if max_profit_upper is None:
            max_profit_upper = breakeven_upper

        return {
            'prob_profit': self.prob_between_prices(breakeven_lower, breakeven_upper),
            'prob_max_profit': self.prob_between_prices(max_profit_lower, max_profit_upper),
            'prob_below_lower': self.prob_below_price(breakeven_lower),
            'prob_above_upper': self.prob_above_price(breakeven_upper)
        }
