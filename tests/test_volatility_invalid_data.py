import pytest
import numpy as np
import pandas as pd
from src.utils.volatility import VolatilityAnalyzer
import math

class TestVolatilityInvalidData:
    """Test handling of invalid and malformed data in volatility analysis"""

    @pytest.fixture
    def invalid_market_data_samples(self):
        """Generate various invalid market data scenarios"""
        return {
            'negative_strikes': pd.DataFrame({
                'strike': [-100, 100, -50],
                'expiry': [0.25, 0.25, 0.25],
                'implied_vol': [0.2, 0.18, 0.22],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'negative_expiry': pd.DataFrame({
                'strike': [90, 100, 110],
                'expiry': [-0.1, 0.25, -0.5],
                'implied_vol': [0.2, 0.18, 0.22],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'negative_vol': pd.DataFrame({
                'strike': [90, 100, 110],
                'expiry': [0.25, 0.25, 0.25],
                'implied_vol': [-0.2, 0.18, -0.1],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'invalid_option_type': pd.DataFrame({
                'strike': [90, 100, 110],
                'expiry': [0.25, 0.25, 0.25],
                'implied_vol': [0.2, 0.18, 0.22],
                'option_type': ['invalid', 'call', 'unknown'],
                'underlying_price': [100] * 3
            }),
            
            'string_values': pd.DataFrame({
                'strike': ['abc', '100', 'xyz'],
                'expiry': [0.25, 'six months', 0.25],
                'implied_vol': [0.2, 'high', 0.22],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'infinity_values': pd.DataFrame({
                'strike': [90, np.inf, 110],
                'expiry': [0.25, 0.25, np.inf],
                'implied_vol': [0.2, np.inf, 0.22],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'nan_values': pd.DataFrame({
                'strike': [90, np.nan, 110],
                'expiry': [0.25, np.nan, 0.25],
                'implied_vol': [0.2, 0.18, np.nan],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'mixed_types': pd.DataFrame({
                'strike': [90, '100', 110.5],
                'expiry': [0.25, '3m', 0.5],
                'implied_vol': [0.2, '20%', 0.22],
                'option_type': ['call'] * 3,
                'underlying_price': [100] * 3
            }),
            
            'special_characters': pd.DataFrame({
                'strike': [90, '100$', '110€'],
                'expiry': [0.25, '3m!', '6m@'],
                'implied_vol': [0.2, '20%', '22%'],
                'option_type': ['call'] * 3,
                'underlying_price': ['100$'] * 3
            })
        }

    def test_negative_values(self, invalid_market_data_samples):
        """Test handling of negative values"""
        for data_type in ['negative_strikes', 'negative_expiry', 'negative_vol']:
            with pytest.raises(ValueError, match=f"Negative values not allowed in {data_type}"):
                VolatilityAnalyzer(invalid_market_data_samples[data_type])

    def test_invalid_option_types(self, invalid_market_data_samples):
        """Test handling of invalid option types"""
        with pytest.raises(ValueError, match="Invalid option type"):
            VolatilityAnalyzer(invalid_market_data_samples['invalid_option_type'])

    def test_string_numeric_values(self, invalid_market_data_samples):
        """Test handling of string values in numeric fields"""
        with pytest.raises(ValueError, match="Non-numeric values found"):
            VolatilityAnalyzer(invalid_market_data_samples['string_values'])

    def test_infinity_values(self, invalid_market_data_samples):
        """Test handling of infinity values"""
        with pytest.raises(ValueError, match="Infinite values not allowed"):
            VolatilityAnalyzer(invalid_market_data_samples['infinity_values'])

    def test_nan_values(self, invalid_market_data_samples):
        """Test handling of NaN values"""
        with pytest.raises(ValueError, match="NaN values not allowed"):
            VolatilityAnalyzer(invalid_market_data_samples['nan_values'])

    def test_mixed_types(self, invalid_market_data_samples):
        """Test handling of mixed data types"""
        with pytest.raises(ValueError, match="Mixed data types not allowed"):
            VolatilityAnalyzer(invalid_market_data_samples['mixed_types'])

    def test_special_characters(self, invalid_market_data_samples):
        """Test handling of special characters"""
        with pytest.raises(ValueError, match="Special characters not allowed"):
            VolatilityAnalyzer(invalid_market_data_samples['special_characters'])

    @pytest.mark.parametrize("invalid_value", [
        math.nan,
        math.inf,
        -math.inf,
        None,
        "",
        " ",
        "N/A",
        "#N/A",
        "NULL",
        "None"
    ])
    def test_common_invalid_values(self, invalid_value):
        """Test handling of common invalid values"""
        invalid_data = pd.DataFrame({
            'strike': [90, invalid_value, 110],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.18, 0.22],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        with pytest.raises(ValueError):
            VolatilityAnalyzer(invalid_data)

    def test_data_type_conversion(self):
        """Test automatic data type conversion for valid but incorrectly typed data"""
        convertible_data = pd.DataFrame({
            'strike': ['90.0', '100.0', '110.0'],  # Strings that can be converted to float
            'expiry': ['0.25', '0.5', '1.0'],
            'implied_vol': ['0.2', '0.18', '0.22'],
            'option_type': ['call'] * 3,
            'underlying_price': ['100.0'] * 3
        })
        
        # This should not raise an error if conversion is handled
        analyzer = VolatilityAnalyzer(convertible_data)
        assert isinstance(analyzer.market_data['strike'].iloc[0], float)

    def test_empty_data(self):
        """Test handling of empty DataFrames"""
        empty_data = pd.DataFrame(columns=[
            'strike', 'expiry', 'implied_vol', 'option_type', 'underlying_price'
        ])
        
        with pytest.raises(ValueError, match="Empty dataset"):
            VolatilityAnalyzer(empty_data)

    def test_duplicate_data(self):
        """Test handling of duplicate entries"""
        duplicate_data = pd.DataFrame({
            'strike': [100, 100, 100],
            'expiry': [0.25, 0.25, 0.25],
            'implied_vol': [0.2, 0.2, 0.2],
            'option_type': ['call'] * 3,
            'underlying_price': [100] * 3
        })
        
        analyzer = VolatilityAnalyzer(duplicate_data)
        assert len(analyzer.market_data) < len(duplicate_data)  # Should remove duplicates 