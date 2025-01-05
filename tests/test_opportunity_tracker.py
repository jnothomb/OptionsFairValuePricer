import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.opportunity_tracker import (
    OpportunityTracker, 
    OpportunityType,
    OpportunityPriority,
    MarketOpportunity
)

class TestOpportunityTracker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = OpportunityTracker()
        self.sample_vol_surface = pd.DataFrame({
            'strike': [95, 100, 105] * 3,
            'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
            'option_type': ['call'] * 9,
            'underlying_price': [100] * 9
        })
        
        self.sample_options_data = pd.DataFrame({
            'strike': [100] * 2,
            'expiry': [0.25] * 2,
            'option_type': ['call', 'put'],
            'price': [5.0, 4.0],
            'underlying_price': [100] * 2
        })
        
        self.sample_portfolio = pd.DataFrame({
            'instrument': ['OPT_100_0.25', 'OPT_105_0.5'],
            'position_value': [1000000, 100000]
        })

    def test_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(len(self.tracker.opportunities), 0)
        self.assertEqual(self.tracker.metrics['total_opportunities'], 0)
        self.assertEqual(self.tracker.metrics['active_opportunities'], 0)
        self.assertEqual(self.tracker.metrics['success_rate'], 0.0)

    def test_add_opportunity(self):
        """Test adding new opportunities"""
        opportunity = MarketOpportunity(
            type=OpportunityType.BUTTERFLY_ARBITRAGE,
            priority=OpportunityPriority.HIGH,
            description="Test opportunity",
            timestamp=datetime.now()
        )
        
        self.tracker.add_opportunity(opportunity)
        
        self.assertEqual(len(self.tracker.opportunities), 1)
        self.assertEqual(self.tracker.metrics['total_opportunities'], 1)
        self.assertEqual(self.tracker.metrics['active_opportunities'], 1)
        self.assertEqual(
            self.tracker.metrics['by_type'][OpportunityType.BUTTERFLY_ARBITRAGE], 
            1
        )

    def test_close_opportunity(self):
        """Test closing opportunities"""
        opportunity = MarketOpportunity(
            type=OpportunityType.BUTTERFLY_ARBITRAGE,
            priority=OpportunityPriority.HIGH,
            description="Test opportunity",
            timestamp=datetime.now()
        )
        
        self.tracker.add_opportunity(opportunity)
        self.tracker.close_opportunity(opportunity, "Success")
        
        self.assertEqual(opportunity.status, "closed")
        self.assertEqual(self.tracker.metrics['active_opportunities'], 0)

    def test_get_active_opportunities(self):
        """Test filtering active opportunities"""
        # Add mixed priority opportunities
        opportunities = [
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description="High priority",
                timestamp=datetime.now()
            ),
            MarketOpportunity(
                type=OpportunityType.CALENDAR_SPREAD,
                priority=OpportunityPriority.MEDIUM,
                description="Medium priority",
                timestamp=datetime.now()
            ),
            MarketOpportunity(
                type=OpportunityType.RISK_CONCENTRATION,
                priority=OpportunityPriority.LOW,
                description="Low priority",
                timestamp=datetime.now()
            )
        ]
        
        for op in opportunities:
            self.tracker.add_opportunity(op)
            
        # Test priority filtering
        high_priority = self.tracker.get_active_opportunities(
            min_priority=OpportunityPriority.HIGH
        )
        self.assertEqual(len(high_priority), 1)  # Only one HIGH priority
        self.assertEqual(high_priority[0].priority, OpportunityPriority.HIGH)
        
        medium_and_above = self.tracker.get_active_opportunities(
            min_priority=OpportunityPriority.MEDIUM
        )
        self.assertEqual(len(medium_and_above), 2)  # HIGH and MEDIUM
        
        # Test type filtering
        butterfly_ops = self.tracker.get_active_opportunities(
            opportunity_type=OpportunityType.BUTTERFLY_ARBITRAGE
        )
        self.assertEqual(len(butterfly_ops), 1)
        self.assertEqual(butterfly_ops[0].type, OpportunityType.BUTTERFLY_ARBITRAGE)
        
        # Test combined filtering
        high_butterfly_ops = self.tracker.get_active_opportunities(
            min_priority=OpportunityPriority.HIGH,
            opportunity_type=OpportunityType.BUTTERFLY_ARBITRAGE
        )
        self.assertEqual(len(high_butterfly_ops), 1)
        self.assertEqual(high_butterfly_ops[0].priority, OpportunityPriority.HIGH)
        self.assertEqual(high_butterfly_ops[0].type, OpportunityType.BUTTERFLY_ARBITRAGE)

    def test_volatility_arbitrage_detection(self):
        """Test volatility arbitrage detection"""
        # Create a surface with clear butterfly arbitrage
        vol_surface = self.sample_vol_surface.copy()
        vol_surface.loc[1, 'implied_vol'] = 0.3  # Create arbitrage opportunity
        
        self.tracker.analyze_volatility_arbitrage(vol_surface)
        
        butterfly_ops = self.tracker.get_active_opportunities(
            opportunity_type=OpportunityType.BUTTERFLY_ARBITRAGE
        )
        self.assertGreater(len(butterfly_ops), 0)

    def test_calendar_spread_detection(self):
        """Test calendar spread detection"""
        # Create a surface with calendar spread opportunity
        vol_surface = self.sample_vol_surface.copy()
        vol_surface.loc[vol_surface['expiry'] == 0.25, 'implied_vol'] = 0.25
        vol_surface.loc[vol_surface['expiry'] == 0.5, 'implied_vol'] = 0.20
        
        self.tracker.analyze_calendar_spreads(vol_surface)
        
        calendar_ops = self.tracker.get_active_opportunities(
            opportunity_type=OpportunityType.CALENDAR_SPREAD
        )
        self.assertGreater(len(calendar_ops), 0)

    def test_put_call_parity_detection(self):
        """Test put-call parity violation detection"""
        # Create data with put-call parity violation
        options_data = self.sample_options_data.copy()
        options_data.loc[0, 'price'] = 7.0  # Create parity violation
        
        self.tracker.analyze_put_call_parity(options_data)
        
        parity_ops = self.tracker.get_active_opportunities(
            opportunity_type=OpportunityType.PUT_CALL_PARITY
        )
        self.assertGreater(len(parity_ops), 0)

    def test_risk_concentration_detection(self):
        """Test risk concentration detection"""
        self.tracker.analyze_risk_concentration(self.sample_portfolio)
        
        concentration_ops = self.tracker.get_active_opportunities(
            opportunity_type=OpportunityType.RISK_CONCENTRATION
        )
        self.assertGreater(len(concentration_ops), 0)

    def test_opportunity_expiry(self):
        """Test opportunity expiry handling"""
        opportunity = MarketOpportunity(
            type=OpportunityType.BUTTERFLY_ARBITRAGE,
            priority=OpportunityPriority.HIGH,
            description="Expiring opportunity",
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(hours=1)
        )
        
        self.tracker.add_opportunity(opportunity)
        self.assertTrue(opportunity.expiry > datetime.now())

    def test_metrics_calculation(self):
        """Test metrics calculation"""
        # Add opportunities with different outcomes
        for i in range(5):
            opportunity = MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description=f"Test opportunity {i}",
                timestamp=datetime.now()
            )
            self.tracker.add_opportunity(opportunity)
            if i < 3:  # Close some opportunities
                self.tracker.close_opportunity(opportunity, "Success")
                
        summary = self.tracker.get_opportunity_summary()
        self.assertEqual(summary['total_count'], 5)
        self.assertEqual(summary['active_count'], 2)

    def test_historical_analysis(self):
        """Test historical analysis"""
        # Add and close opportunities
        for i in range(3):
            opportunity = MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description=f"Historical opportunity {i}",
                timestamp=datetime.now() - timedelta(days=i)
            )
            self.tracker.add_opportunity(opportunity)
            self.tracker.close_opportunity(opportunity, "Success")
            
        history = self.tracker.get_historical_analysis()
        self.assertEqual(len(history), 3)
        self.assertTrue('timestamp' in history.columns)
        self.assertTrue('status' in history.columns)

    def test_opportunity_report_generation(self):
        """Test report generation"""
        # Add mixed opportunities
        opportunities = [
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description="High priority report test",
                timestamp=datetime.now(),
                action_items=["Test action 1", "Test action 2"]
            ),
            MarketOpportunity(
                type=OpportunityType.CALENDAR_SPREAD,
                priority=OpportunityPriority.MEDIUM,
                description="Medium priority report test",
                timestamp=datetime.now()
            )
        ]
        
        for op in opportunities:
            self.tracker.add_opportunity(op)
            
        report = self.tracker.generate_opportunity_report()
        self.assertIsInstance(report, str)
        self.assertIn("High Priority", report)
        self.assertIn("Test action 1", report)

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty data frames
        empty_surface = pd.DataFrame(columns=self.sample_vol_surface.columns)
        empty_portfolio = pd.DataFrame(columns=self.sample_portfolio.columns)
        
        # These should not raise exceptions
        self.tracker.analyze_volatility_arbitrage(empty_surface)
        self.tracker.analyze_calendar_spreads(empty_surface)
        self.tracker.analyze_risk_concentration(empty_portfolio)
        
        # Test invalid opportunity type
        with self.assertRaises(ValueError):
            MarketOpportunity(
                type="INVALID_TYPE",  # Invalid type
                priority=OpportunityPriority.HIGH,
                description="Invalid type test",
                timestamp=datetime.now()
            )
            
        # Test invalid priority
        with self.assertRaises(ValueError):
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority="HIGH",  # Invalid priority format
                description="Invalid priority test",
                timestamp=datetime.now()
            )
            
        # Test invalid timestamp
        with self.assertRaises(ValueError):
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description="Invalid timestamp test",
                timestamp="2023-01-01"  # Invalid timestamp format
            )
            
        # Test invalid expiry
        with self.assertRaises(ValueError):
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description="Invalid expiry test",
                timestamp=datetime.now(),
                expiry="2024-01-01"  # Invalid expiry format
            )
            
        # Test empty description
        with self.assertRaises(ValueError):
            MarketOpportunity(
                type=OpportunityType.BUTTERFLY_ARBITRAGE,
                priority=OpportunityPriority.HIGH,
                description="",  # Empty description
                timestamp=datetime.now()
            )

if __name__ == '__main__':
    unittest.main() 