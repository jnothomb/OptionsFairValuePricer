import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from src.utils.greeks import GreeksCalculator
from src.utils.greeks import PortfolioGreeks

class GreeksVisualizer:
    """
    Visualization tools for option Greeks and risk metrics
    
    Provides various plotting capabilities:
    - 3D surface plots for Greeks vs price and time
    - Sensitivity analysis plots
    - Risk dashboard visualization
    
    Uses matplotlib and seaborn for high-quality visualizations
    """
    
    def __init__(self, style='darkgrid'):
        sns.set_style(style)
        
    def plot_greeks_surface(self, spot_range: np.array, time_range: np.array, 
                           calculator: 'GreeksCalculator', greek: str):
        """
        Create a 3D surface plot showing how a Greek varies with price and time
        
        Args:
            spot_range: Array of underlying prices to plot
            time_range: Array of time points to plot
            calculator: Instance of GreeksCalculator
            greek: Which Greek to plot ('delta', 'gamma', etc.)
            
        Returns:
            matplotlib figure object
            
        Note: Useful for visualizing term structure of Greeks
        """
        X, Y = np.meshgrid(spot_range, time_range)
        Z = np.zeros_like(X)
        
        for i, t in enumerate(time_range):
            for j, s in enumerate(spot_range):
                calc = calculator(spot_price=s, time_to_expiry=t)
                greeks = calc.calculate_greeks()
                Z[i,j] = getattr(greeks, greek.lower())
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        plt.colorbar(surface)
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel(greek.capitalize())
        plt.title(f'{greek.capitalize()} Surface')
        
        return fig

    def plot_greeks_sensitivity(self, strategy_positions: List[Dict], 
                              price_range: np.array):
        """Plot Greeks sensitivity to price changes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        greeks = ['Delta', 'Gamma', 'Theta', 'Vega']
        
        for greek, ax in zip(greeks, axes.flat):
            values = []
            for price in price_range:
                positions = self._update_positions_price(strategy_positions, price)
                portfolio = PortfolioGreeks(positions)
                greeks_values = portfolio.calculate_portfolio_greeks()
                values.append(getattr(greeks_values, greek.lower()))
            
            ax.plot(price_range, values)
            ax.set_title(f'{greek} Sensitivity')
            ax.set_xlabel('Stock Price')
            ax.set_ylabel(greek)
            ax.grid(True)
        
        plt.tight_layout()
        return fig

    def plot_risk_dashboard(self, strategy_positions: List[Dict], 
                          thresholds: Dict):
        """Create a risk dashboard with Greek values and thresholds"""
        portfolio = PortfolioGreeks(strategy_positions)
        greeks = portfolio.calculate_portfolio_greeks()
        risk_metrics = portfolio.risk_metrics()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        
        bars = ax.barh(metrics, values)
        
        # Add threshold lines
        for metric, threshold in thresholds.items():
            if metric in risk_metrics:
                ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title('Risk Metrics Dashboard')
        plt.tight_layout()
        return fig