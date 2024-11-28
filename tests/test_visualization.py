import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from src.utils.volatility_viz import VolatilityVisualizer
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class TestVisualization:
    """Test suite for visualization components"""

    @pytest.fixture
    def vol_visualizer(self):
        """Initialize volatility visualizer"""
        return VolatilityVisualizer(style='darkgrid')

    @pytest.fixture
    def sample_surface_data(self):
        """Generate sample volatility surface data"""
        strikes = np.linspace(90, 110, 21)
        expiries = np.array([0.25, 0.5, 1.0])
        implied_vols = np.random.normal(0.2, 0.02, size=(len(expiries), len(strikes)))
        return {
            'strikes': strikes,
            'expiries': expiries,
            'implied_vols': implied_vols,
            'forward_price': 100,
            'timestamp': pd.Timestamp.now(),
            'quality_metrics': {'rmse': 0.001}
        }

    def test_vol_surface_plot(self, vol_visualizer, sample_surface_data):
        """Test volatility surface plotting"""
        fig = vol_visualizer.plot_volatility_surface(**sample_surface_data)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert isinstance(fig.axes[0], Axes3D)
        
        # Check plot elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Strike'
        assert ax.get_ylabel() == 'Time to Expiry'
        assert ax.get_zlabel() == 'Implied Volatility'

    def test_vol_smile_plot(self, vol_visualizer, sample_surface_data):
        """Test volatility smile plotting"""
        fig = vol_visualizer.plot_volatility_smile(
            strikes=sample_surface_data['strikes'],
            implied_vols=sample_surface_data['implied_vols'][0],
            expiry=sample_surface_data['expiries'][0]
        )
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        
        # Check plot elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Strike'
        assert ax.get_ylabel() == 'Implied Volatility'
        assert len(ax.lines) >= 1  # At least one line plotted

    @pytest.mark.parametrize("style", ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'])
    def test_visualization_styles(self, style):
        """Test different visualization styles"""
        vol_viz = VolatilityVisualizer(style=style)
        
        assert vol_viz.style == style

    def test_plot_saving(self, vol_visualizer, sample_surface_data, tmp_path):
        """Test plot saving functionality"""
        # Test volatility surface save
        surface_path = tmp_path / "vol_surface.png"
        vol_visualizer.plot_volatility_surface(
            **sample_surface_data,
            save_path=str(surface_path)
        )
        assert surface_path.exists()
        
    def test_plot_customization(self, vol_visualizer, sample_surface_data):
        """Test plot customization options"""
        # Test volatility surface customization
        custom_surface = vol_visualizer.plot_volatility_surface(
            **sample_surface_data,
            title="Custom Surface Plot",
            colormap="viridis",
            alpha=0.7,
            figsize=(12, 8)
        )
        assert custom_surface.axes[0].get_title() == "Custom Surface Plot"

    def test_error_handling(self, vol_visualizer):
        """Test error handling in visualization"""
        # Test invalid data
        with pytest.raises(ValueError):
            vol_visualizer.plot_volatility_surface(
                strikes=np.array([]),  # Empty array
                expiries=np.array([0.5]),
                implied_vols=np.array([[0.2]]),
                forward_price=100,
                timestamp=pd.Timestamp.now(),
                quality_metrics={}
            )

    def test_interactive_features(self, vol_visualizer, sample_surface_data):
        """Test interactive plot features"""
        fig = vol_visualizer.plot_volatility_surface(
            **sample_surface_data,
            interactive=True
        )
        
        # Check interactive elements
        assert hasattr(fig, '_button_pressed')  # Matplotlib interactive attribute