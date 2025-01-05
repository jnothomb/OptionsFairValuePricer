from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class OpportunityType(Enum):
    """Classification of market opportunities"""
    VOLATILITY_ARBITRAGE = "Volatility Arbitrage"
    CALENDAR_SPREAD = "Calendar Spread"
    BUTTERFLY_ARBITRAGE = "Butterfly Arbitrage"
    PUT_CALL_PARITY = "Put-Call Parity"
    VOLATILITY_REGIME = "Volatility Regime"
    MARKET_MISPRICING = "Market Mispricing"
    RISK_CONCENTRATION = "Risk Concentration"
    MARKET_DISLOCATION = "Market Dislocation"

class OpportunityPriority(Enum):
    """Priority levels for market opportunities"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class MarketOpportunity:
    """
    Represents a detected market opportunity
    
    Attributes:
        type: Type of opportunity
        priority: Priority level
        description: Detailed description
        timestamp: When opportunity was detected
        expiry: Optional expiry time for opportunity
        metrics: Relevant quantitative metrics
        instruments: Affected instruments
        action_items: Suggested actions
        status: Current status (open/closed)
    """
    type: OpportunityType
    priority: OpportunityPriority
    description: str
    timestamp: datetime
    expiry: Optional[datetime] = None
    metrics: Dict[str, float] = None
    instruments: List[str] = None
    action_items: List[str] = None
    status: str = "open"
    
    def __post_init__(self):
        """Validate opportunity attributes after initialization"""
        if not isinstance(self.type, OpportunityType):
            raise ValueError(f"Invalid opportunity type: {self.type}")
        if not isinstance(self.priority, OpportunityPriority):
            raise ValueError(f"Invalid priority level: {self.priority}")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
        if self.expiry and not isinstance(self.expiry, datetime):
            raise ValueError("Expiry must be a datetime object")
        if not self.description or not self.description.strip():
            raise ValueError("Description cannot be empty")

class OpportunityTracker:
    """
    Centralized system for tracking and managing market opportunities
    
    Features:
    1. Opportunity Detection
       - Volatility surface anomalies
       - Statistical arbitrage signals
       - Risk concentration alerts
       - Market regime changes
    
    2. Priority Management
       - Dynamic priority assignment
       - Expiry tracking
       - Action item generation
    
    3. Reporting & Analytics
       - Opportunity dashboard
       - Historical analysis
       - Success rate tracking
    """
    
    def __init__(self):
        """Initialize opportunity tracking system"""
        self.opportunities = []
        self.history = pd.DataFrame()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize tracking metrics"""
        self.metrics = {
            'total_opportunities': 0,
            'active_opportunities': 0,
            'success_rate': 0.0,
            'avg_duration': pd.Timedelta(0),
            'by_type': {},
            'by_priority': {}
        }
    
    def add_opportunity(self, opportunity: MarketOpportunity) -> None:
        """
        Add new market opportunity to tracking system
        
        Args:
            opportunity: MarketOpportunity instance to track
        """
        # Add opportunity to active list
        self.opportunities.append(opportunity)
        
        # Update metrics
        self.metrics['total_opportunities'] += 1
        self.metrics['active_opportunities'] += 1
        
        # Update type and priority counts
        self.metrics['by_type'][opportunity.type] = (
            self.metrics['by_type'].get(opportunity.type, 0) + 1
        )
        self.metrics['by_priority'][opportunity.priority] = (
            self.metrics['by_priority'].get(opportunity.priority, 0) + 1
        )
        
        # Log to history
        self._log_to_history(opportunity)
    
    def get_active_opportunities(self, 
                               min_priority: OpportunityPriority = None,
                               opportunity_type: OpportunityType = None) -> List[MarketOpportunity]:
        """
        Get filtered list of active opportunities
        
        Args:
            min_priority: Minimum priority level to include
            opportunity_type: Specific type to filter for
            
        Returns:
            List of matching active opportunities
        """
        # Define priority values for comparison
        priority_values = {
            OpportunityPriority.LOW: 0,
            OpportunityPriority.MEDIUM: 1,
            OpportunityPriority.HIGH: 2
        }
        
        opportunities = [op for op in self.opportunities if op.status == "open"]
        
        if min_priority:
            min_value = priority_values[min_priority]
            opportunities = [
                op for op in opportunities 
                if priority_values[op.priority] >= min_value
            ]
            
        if opportunity_type:
            opportunities = [
                op for op in opportunities 
                if op.type == opportunity_type
            ]
            
        # Define sort key function with access to priority_values
        def sort_key(x):
            return (priority_values[x.priority], x.timestamp)
            
        return sorted(opportunities, key=sort_key, reverse=True)
    
    def close_opportunity(self, opportunity: MarketOpportunity, 
                         outcome: str = None) -> None:
        """
        Mark opportunity as closed with optional outcome
        
        Args:
            opportunity: Opportunity to close
            outcome: Optional outcome description
        """
        if opportunity in self.opportunities:
            opportunity.status = "closed"
            self.metrics['active_opportunities'] -= 1
            
            # Update history
            mask = (self.history['timestamp'] == opportunity.timestamp) & \
                  (self.history['type'] == opportunity.type.value)
            self.history.loc[mask, 'status'] = 'closed'
            if outcome:
                self.history.loc[mask, 'outcome'] = outcome
    
    def get_opportunity_summary(self) -> Dict:
        """
        Get summary statistics of tracked opportunities
        
        Returns:
            Dict containing:
            - Active counts by type/priority
            - Success rates
            - Average durations
            - Historical patterns
        """
        return {
            'active_count': self.metrics['active_opportunities'],
            'total_count': self.metrics['total_opportunities'],
            'by_type': self.metrics['by_type'],
            'by_priority': self.metrics['by_priority'],
            'success_rate': self.metrics['success_rate'],
            'avg_duration': self.metrics['avg_duration']
        }
    
    def analyze_volatility_arbitrage(self, vol_surface: pd.DataFrame) -> None:
        """
        Detect volatility arbitrage opportunities
        
        Args:
            vol_surface: Volatility surface data
        """
        # Butterfly arbitrage detection
        for expiry in vol_surface['expiry'].unique():
            expiry_data = vol_surface[vol_surface['expiry'] == expiry]
            strikes = sorted(expiry_data['strike'].unique())
            
            for i in range(1, len(strikes)-1):
                k1, k2, k3 = strikes[i-1:i+2]
                v1 = expiry_data[expiry_data['strike'] == k1]['implied_vol'].iloc[0]
                v2 = expiry_data[expiry_data['strike'] == k2]['implied_vol'].iloc[0]
                v3 = expiry_data[expiry_data['strike'] == k3]['implied_vol'].iloc[0]
                
                # Check butterfly arbitrage
                butterfly_violation = v2 > (k3-k2)/(k3-k1)*v1 + (k2-k1)/(k3-k1)*v3
                if butterfly_violation:
                    self.add_opportunity(MarketOpportunity(
                        type=OpportunityType.BUTTERFLY_ARBITRAGE,
                        priority=OpportunityPriority.HIGH,
                        description=f"Butterfly arbitrage at expiry {expiry}",
                        timestamp=datetime.now(),
                        expiry=datetime.now() + pd.Timedelta(days=1),
                        metrics={
                            'strike1': k1, 'vol1': v1,
                            'strike2': k2, 'vol2': v2,
                            'strike3': k3, 'vol3': v3
                        },
                        instruments=[f"OPT_{k2}_{expiry}"],
                        action_items=[
                            f"Sell {k2} strike, Buy {k1} and {k3} strikes",
                            "Monitor execution prices",
                            "Set stop-loss at 50% of theoretical edge"
                        ]
                    ))
    
    def analyze_calendar_spreads(self, vol_surface: pd.DataFrame) -> None:
        """
        Detect calendar spread opportunities
        
        Args:
            vol_surface: Volatility surface data
        """
        for strike in vol_surface['strike'].unique():
            strike_data = vol_surface[vol_surface['strike'] == strike]
            expiries = sorted(strike_data['expiry'].unique())
            vols = [
                strike_data[strike_data['expiry'] == t]['implied_vol'].iloc[0] 
                for t in expiries
            ]
            
            for i in range(len(expiries)-1):
                if vols[i] > vols[i+1]:
                    self.add_opportunity(MarketOpportunity(
                        type=OpportunityType.CALENDAR_SPREAD,
                        priority=OpportunityPriority.MEDIUM,
                        description=f"Calendar spread at strike {strike}",
                        timestamp=datetime.now(),
                        expiry=datetime.now() + pd.Timedelta(days=1),
                        metrics={
                            'strike': strike,
                            'near_expiry': expiries[i],
                            'far_expiry': expiries[i+1],
                            'near_vol': vols[i],
                            'far_vol': vols[i+1]
                        },
                        instruments=[
                            f"OPT_{strike}_{expiries[i]}",
                            f"OPT_{strike}_{expiries[i+1]}"
                        ],
                        action_items=[
                            f"Sell {expiries[i]} expiry, Buy {expiries[i+1]} expiry",
                            "Check historical vol term structure",
                            "Monitor calendar spread ratio"
                        ]
                    ))
    
    def analyze_put_call_parity(self, options_data: pd.DataFrame) -> None:
        """
        Detect put-call parity violations
        
        Args:
            options_data: Options market data
        """
        for expiry in options_data['expiry'].unique():
            expiry_data = options_data[options_data['expiry'] == expiry]
            
            # Group by strike
            for strike in expiry_data['strike'].unique():
                strike_data = expiry_data[expiry_data['strike'] == strike]
                
                # Get call and put prices
                call_data = strike_data[strike_data['option_type'] == 'call']
                put_data = strike_data[strike_data['option_type'] == 'put']
                
                if len(call_data) > 0 and len(put_data) > 0:
                    call_price = call_data['price'].iloc[0]
                    put_price = put_data['price'].iloc[0]
                    spot = options_data['underlying_price'].iloc[0]
                    r = 0.02  # Risk-free rate (should be parameter)
                    
                    # Check put-call parity
                    parity_diff = abs(
                        call_price - put_price - 
                        spot + strike * np.exp(-r * expiry)
                    )
                    
                    if parity_diff > 0.1:  # Threshold for significance
                        self.add_opportunity(MarketOpportunity(
                            type=OpportunityType.PUT_CALL_PARITY,
                            priority=OpportunityPriority.HIGH,
                            description=f"Put-call parity violation at strike {strike}",
                            timestamp=datetime.now(),
                            expiry=datetime.now() + pd.Timedelta(days=1),
                            metrics={
                                'strike': strike,
                                'expiry': expiry,
                                'call_price': call_price,
                                'put_price': put_price,
                                'parity_diff': parity_diff
                            },
                            instruments=[
                                f"CALL_{strike}_{expiry}",
                                f"PUT_{strike}_{expiry}"
                            ],
                            action_items=[
                                "Verify bid-ask spreads",
                                "Check borrow rates",
                                "Validate dividend assumptions"
                            ]
                        ))
    
    def analyze_risk_concentration(self, portfolio: pd.DataFrame) -> None:
        """
        Detect portfolio risk concentrations
        
        Args:
            portfolio: Portfolio position data
        """
        # Calculate position concentrations
        total_value = portfolio['position_value'].abs().sum()
        
        for _, position in portfolio.iterrows():
            concentration = abs(position['position_value']) / total_value
            
            if concentration > 0.2:  # 20% threshold
                self.add_opportunity(MarketOpportunity(
                    type=OpportunityType.RISK_CONCENTRATION,
                    priority=OpportunityPriority.MEDIUM,
                    description=f"Large position concentration",
                    timestamp=datetime.now(),
                    metrics={
                        'instrument': position['instrument'],
                        'concentration': concentration,
                        'position_value': position['position_value']
                    },
                    instruments=[position['instrument']],
                    action_items=[
                        "Review position sizing",
                        "Check correlation with other positions",
                        "Consider partial reduction"
                    ]
                ))
    
    def _log_to_history(self, opportunity: MarketOpportunity) -> None:
        """Log opportunity to historical record"""
        entry = {
            'timestamp': opportunity.timestamp,
            'type': opportunity.type.value,
            'priority': opportunity.priority.value,
            'description': opportunity.description,
            'status': opportunity.status
        }
        
        self.history = pd.concat([
            self.history,
            pd.DataFrame([entry])
        ], ignore_index=True)
    
    def get_historical_analysis(self) -> pd.DataFrame:
        """
        Get historical opportunity analysis
        
        Returns:
            DataFrame with historical opportunity data and outcomes
        """
        return self.history.copy()
    
    def generate_opportunity_report(self) -> str:
        """
        Generate formatted report of current opportunities
        
        Returns:
            Formatted string containing opportunity report
        """
        active = self.get_active_opportunities()
        
        report = ["Market Opportunity Report", "=" * 25, ""]
        
        for priority in OpportunityPriority:
            priority_ops = [op for op in active if op.priority == priority]
            if priority_ops:
                report.append(f"\n{priority.value} Priority:")
                report.append("-" * 15)
                
                for op in priority_ops:
                    report.extend([
                        f"Type: {op.type.value}",
                        f"Description: {op.description}",
                        f"Detected: {op.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        "Action Items:",
                        *[f"- {item}" for item in (op.action_items or [])],
                        ""
                    ])
        
        return "\n".join(report) 