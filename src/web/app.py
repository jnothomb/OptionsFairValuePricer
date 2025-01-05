from flask import Flask, render_template
from src.utils.opportunity_tracker import OpportunityTracker, OpportunityType
import pandas as pd

app = Flask(__name__)
opportunity_tracker = OpportunityTracker()

@app.route('/')
@app.route('/opportunities')
def opportunities_dashboard():
    """Render the opportunities dashboard"""
    # Get active opportunities and summary
    opportunities = opportunity_tracker.get_active_opportunities()
    summary = opportunity_tracker.get_opportunity_summary()
    
    # Get all opportunity types for the filter
    opportunity_types = [t.value for t in OpportunityType]
    
    return render_template(
        'opportunities.html',
        opportunities=opportunities,
        summary=summary,
        opportunity_types=opportunity_types
    )

@app.route('/api/opportunities/history')
def opportunity_history():
    """Get historical opportunity data"""
    history = opportunity_tracker.get_historical_analysis()
    return history.to_json(orient='records', date_format='iso')

@app.route('/api/opportunities/summary')
def opportunity_summary():
    """Get opportunity summary statistics"""
    summary = opportunity_tracker.get_opportunity_summary()
    return summary

def run_opportunity_analysis():
    """
    Run opportunity analysis on market data
    This should be called periodically or when market data updates
    """
    # Example market data (replace with actual data source)
    vol_surface = pd.DataFrame({
        'strike': [95, 100, 105] * 3,
        'expiry': [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'implied_vol': [0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.25, 0.23, 0.27],
        'option_type': ['call'] * 9,
        'underlying_price': [100] * 9
    })
    
    # Run analysis
    opportunity_tracker.analyze_volatility_arbitrage(vol_surface)
    opportunity_tracker.analyze_calendar_spreads(vol_surface)
    opportunity_tracker.analyze_put_call_parity(vol_surface)

if __name__ == '__main__':
    # Run initial analysis
    run_opportunity_analysis()
    
    # Start Flask app
    app.run(debug=True) 