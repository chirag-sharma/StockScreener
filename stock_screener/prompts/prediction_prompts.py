"""
Prediction Analysis Prompts
===========================

Contains prompts for various prediction and forecasting tasks.
These prompts are designed for use with AI models in price prediction,
technical analysis, and fundamental forecasting scenarios.
"""

from typing import Dict, Any, List
from .analysis_prompts import BasePrompt


class PricePredictionPrompt(BasePrompt):
    """
    General prompt for price prediction analysis and interpretation.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'current_price', 'predictions']
    
    def generate(self, symbol: str, current_price: float, 
                predictions: Dict[str, Any], time_horizon: str = "30 days") -> str:
        """Generate price prediction analysis prompt"""
        
        return f"""
Analyze the AI price prediction results for {symbol}:

Current Price: ₹{current_price:.2f}
Time Horizon: {time_horizon}
Prediction Data: {predictions}

Provide interpretation of the prediction results focusing on:

1. PREDICTION CONFIDENCE: How reliable are these predictions?
2. RISK-REWARD ANALYSIS: What's the potential upside/downside?
3. KEY DRIVERS: What factors likely influence this prediction?
4. MARKET CONTEXT: How do these predictions fit current market conditions?
5. ACTIONABLE INSIGHTS: What should investors consider?

Respond in structured format:
CONFIDENCE_ASSESSMENT: [High/Medium/Low]
EXPECTED_RETURN: [Percentage expected return]
RISK_LEVEL: [Low/Medium/High/Very High]
KEY_FACTORS: [Main factors driving the prediction]
RECOMMENDATION: [Action based on prediction analysis]
REASONING: [2-3 sentence explanation of the recommendation]
        """


class TechnicalAnalysisPrompt(BasePrompt):
    """
    Prompt for technical analysis interpretation and insights.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'technical_indicators']
    
    def generate(self, symbol: str, technical_indicators: Dict[str, Any], 
                price_data: List[float] = None) -> str:
        """Generate technical analysis prompt"""
        
        indicators_summary = []
        for indicator, value in technical_indicators.items():
            indicators_summary.append(f"- {indicator}: {value}")
        
        price_context = ""
        if price_data:
            price_context = f"\nRecent Price Movement: {price_data[-10:]}"  # Last 10 data points
        
        return f"""
Technical Analysis for {symbol}

Technical Indicators:
{chr(10).join(indicators_summary)}{price_context}

Provide technical analysis covering:

1. TREND ANALYSIS: What's the current trend direction?
2. MOMENTUM: Is momentum building or weakening?
3. SUPPORT/RESISTANCE: Key levels to watch
4. SIGNALS: Buy/sell signals from indicators
5. SHORT-TERM OUTLOOK: Next 1-4 weeks
6. RISK MANAGEMENT: Stop-loss and target levels

Format response as:
TREND: [Bullish/Bearish/Sideways]
MOMENTUM: [Strong/Moderate/Weak/Negative]
SIGNAL_STRENGTH: [Strong/Moderate/Weak]
ENTRY_LEVEL: [Suggested entry price]
TARGET_LEVEL: [Price target]
STOP_LOSS: [Risk management level]
TECHNICAL_SUMMARY: [2-3 sentence technical outlook]
        """


class FundamentalAnalysisPrompt(BasePrompt):
    """
    Prompt for fundamental analysis and valuation insights.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'financial_metrics', 'industry_data']
    
    def generate(self, symbol: str, financial_metrics: Dict[str, Any], 
                industry_data: Dict[str, Any] = None) -> str:
        """Generate fundamental analysis prompt"""
        
        metrics_summary = []
        for metric, value in financial_metrics.items():
            metrics_summary.append(f"- {metric}: {value}")
        
        industry_context = ""
        if industry_data:
            industry_summary = []
            for metric, value in industry_data.items():
                industry_summary.append(f"- {metric}: {value}")
            industry_context = f"""

Industry Benchmarks:
{chr(10).join(industry_summary)}"""
        
        return f"""
Fundamental Analysis for {symbol}

Company Financial Metrics:
{chr(10).join(metrics_summary)}{industry_context}

Conduct fundamental analysis covering:

1. VALUATION ASSESSMENT: Is the stock fairly valued?
2. FINANCIAL STRENGTH: Balance sheet and cash flow health
3. PROFITABILITY TRENDS: Revenue and earnings quality
4. GROWTH PROSPECTS: Future growth potential
5. COMPETITIVE POSITION: Industry standing
6. RISK FACTORS: Key risks to monitor

Provide analysis in format:
INTRINSIC_VALUE: [Estimated fair value]
CURRENT_VALUATION: [Undervalued/Fairly Valued/Overvalued]
FINANCIAL_HEALTH: [Excellent/Good/Fair/Poor]
GROWTH_POTENTIAL: [High/Medium/Low]
COMPETITIVE_MOAT: [Wide/Narrow/None]
INVESTMENT_HORIZON: [Long-term/Medium-term/Short-term suitable]
FUNDAMENTAL_SUMMARY: [3-4 sentence fundamental outlook]
        """


class MultiPeriodPredictionPrompt(BasePrompt):
    """
    Prompt for analyzing multi-period prediction results.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'multi_period_data']
    
    def generate(self, symbol: str, multi_period_data: Dict[str, Dict], 
                current_price: float = None) -> str:
        """Generate multi-period prediction analysis prompt"""
        
        periods_summary = []
        for period, prediction_data in multi_period_data.items():
            if isinstance(prediction_data, dict) and 'predicted_price' in prediction_data:
                price = prediction_data['predicted_price']
                confidence = prediction_data.get('confidence', 0)
                change_pct = ""
                if current_price:
                    change_pct = f" ({((price/current_price - 1) * 100):+.1f}%)"
                periods_summary.append(f"- {period}: ₹{price:.2f}{change_pct} (Confidence: {confidence:.1%})")
        
        current_context = f"Current Price: ₹{current_price:.2f}\n" if current_price else ""
        
        return f"""
Multi-Period Prediction Analysis for {symbol}

{current_context}
Predictions Across Time Horizons:
{chr(10).join(periods_summary)}

Analyze the multi-period predictions focusing on:

1. CONSISTENCY: Are predictions consistent across time periods?
2. TREND DIRECTION: What's the overall price trajectory?
3. CONFIDENCE PATTERNS: How does confidence change over time?
4. INVESTMENT TIMING: What's the optimal investment horizon?
5. RISK EVOLUTION: How does risk change across periods?

Provide structured analysis:
OVERALL_TREND: [Bullish/Bearish/Sideways]
BEST_HORIZON: [Optimal investment time frame]
CONSISTENCY_SCORE: [High/Medium/Low]
CONFIDENCE_TREND: [Increasing/Stable/Decreasing over time]
RECOMMENDED_STRATEGY: [Investment approach based on multi-period view]
TIMELINE_ANALYSIS: [Key insights about timing and horizons]
        """


class RiskAssessmentPrompt(BasePrompt):
    """
    Prompt for comprehensive risk assessment based on various factors.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'risk_factors']
    
    def generate(self, symbol: str, risk_factors: Dict[str, Any], 
                market_conditions: str = None) -> str:
        """Generate risk assessment prompt"""
        
        risk_summary = []
        for risk_type, risk_data in risk_factors.items():
            risk_summary.append(f"- {risk_type}: {risk_data}")
        
        market_context = ""
        if market_conditions:
            market_context = f"\nCurrent Market Conditions: {market_conditions}"
        
        return f"""
Comprehensive Risk Assessment for {symbol}

Identified Risk Factors:
{chr(10).join(risk_summary)}{market_context}

Conduct thorough risk analysis covering:

1. FINANCIAL RISKS: Debt, liquidity, profitability concerns
2. MARKET RISKS: Sector, economic, systematic risks
3. OPERATIONAL RISKS: Business model, competition, management
4. REGULATORY RISKS: Policy changes, compliance issues
5. RISK MITIGATION: How company manages these risks
6. OVERALL RISK PROFILE: Investment suitability

Provide risk assessment in format:
OVERALL_RISK: [Low/Medium/High/Very High]
PRIMARY_RISKS: [Top 2-3 risk factors]
RISK_TREND: [Increasing/Stable/Decreasing]
MITIGATION_QUALITY: [Strong/Adequate/Weak]
INVESTOR_SUITABILITY: [Conservative/Moderate/Aggressive investors]
RISK_SUMMARY: [2-3 sentence risk profile summary]
RECOMMENDED_POSITION_SIZE: [Small/Medium/Large based on risk]
        """
