import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.utils.volatility import VolatilitySurface

class VolatilityVisualizer:
    """
    Visualization tools for volatility analysis
    
    Provides comprehensive plotting capabilities for:
    - Volatility surfaces
    - Volatility smiles
    - Term structure
    - Regime analysis
    - Forecast visualization
    
    Features:
    - Interactive 3D plots
    - Multiple plot styles
    - Customizable aesthetics
    - Export capabilities
    """
    
    def __init__(self, style: str = 'darkgrid'):
        """
        Initialize visualizer with specified style
        
        Args:
            style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        """
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_volatility_surface(self, surface: 'VolatilitySurface', 
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 3D visualization of volatility surface
        
        Args:
            surface: VolatilitySurface object containing surface data
            title: Optional title for the plot
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Note: Uses advanced 3D plotting with:
            - Interactive rotation
            - Color mapping for volatility levels
            - Strike/expiry grid lines
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid
        X, Y = np.meshgrid(surface.strikes, surface.expiries)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, surface.implied_vols, 
                             cmap='viridis', 
                             linewidth=0.5, 
                             antialiased=True)
        
        # Customize plot
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel('Implied Volatility')
        if title:
            ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, label='Implied Volatility')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_volatility_smile(self, 
                            strikes: np.ndarray,
                            implied_vols: np.ndarray,
                            expiry: float,
                            fit_curve: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot volatility smile for specific expiry
        
        Args:
            strikes: Array of strike prices
            implied_vols: Array of implied volatilities
            expiry: Time to expiry
            fit_curve: Optional fitted smile curve
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Note: Shows both market data and fitted curve (if provided)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot market data points
        ax.scatter(strikes, implied_vols, 
                  color=self.colors[0], 
                  label='Market Data',
                  alpha=0.6)
        
        # Plot fitted curve if provided
        if fit_curve is not None:
            ax.plot(strikes, fit_curve, 
                   color=self.colors[1], 
                   label='Fitted Curve',
                   linewidth=2)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Volatility Smile (T = {expiry:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_term_structure(self, 
                          term_structure: Dict[float, float],
                          confidence_intervals: Optional[Dict] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize volatility term structure
        
        Args:
            term_structure: Dictionary of {expiry: implied_vol}
            confidence_intervals: Optional confidence intervals
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Note: Includes optional confidence intervals and annotations
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        expiries = list(term_structure.keys())
        vols = list(term_structure.values())
        
        # Plot main term structure
        ax.plot(expiries, vols, 
                color=self.colors[0], 
                marker='o',
                linewidth=2,
                label='Term Structure')
        
        # Add confidence intervals if provided
        if confidence_intervals:
            upper = confidence_intervals['upper']
            lower = confidence_intervals['lower']
            ax.fill_between(expiries, lower, upper, 
                          color=self.colors[0], 
                          alpha=0.2,
                          label='95% Confidence')
        
        ax.set_xlabel('Time to Expiry')
        ax.set_ylabel('Implied Volatility')
        ax.set_title('Volatility Term Structure')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_regime_analysis(self,
                           historical_vols: pd.Series,
                           regime_changes: pd.Series,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize volatility regime changes over time
        
        Args:
            historical_vols: Time series of historical volatilities
            regime_changes: Series indicating regime changes
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Note: Shows volatility levels with regime classifications
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical volatility
        ax.plot(historical_vols.index, historical_vols.values,
                color=self.colors[0],
                label='Historical Volatility',
                alpha=0.7)
        
        # Add regime backgrounds
        for regime in regime_changes.unique():
            mask = regime_changes == regime
            ax.fill_between(historical_vols.index, 
                          historical_vols.min(), 
                          historical_vols.max(),
                          where=mask,
                          alpha=0.2,
                          label=f'Regime: {regime}')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.set_title('Volatility Regimes Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig 