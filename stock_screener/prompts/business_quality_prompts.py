"""
Business Quality Analysis Prompts
=================================

Contains prompts specifically focused on evaluating business quality,
consistency, and performance trends over time.
"""

from typing import List, Dict, Any
from .analysis_prompts import BasePrompt


class BusinessQualityAnalysisPrompt(BasePrompt):
    """
    Prompt for analyzing business quality based on historical performance trends.
    Focuses on consistency, improvement, or decline in key business metrics.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'roe_trend', 'margin_trend']
    
    def generate(self, symbol: str, roe_trend: List[float], margin_trend: List[float]) -> str:
        """Generate business quality analysis prompt"""
        
        return f"""
You are a value investing analyst.

A stock {symbol} has the following historical performance:

ROE (%): {roe_trend}
Operating Margin (%): {margin_trend}

Analyze the business quality. Is it consistent, improving, or declining? Give a short analysis and rate the business quality as one of: High, Medium, Low.
Respond in JSON format with fields:
- "Business Quality Score"
- "AI Business Commentary"
        """


class ComprehensiveBusinessQualityPrompt(BasePrompt):
    """
    Enhanced business quality prompt that includes multiple metrics and deeper analysis.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'metrics_trends']
    
    def generate(self, symbol: str, metrics_trends: Dict[str, List[float]], 
                years: List[int] = None) -> str:
        """Generate comprehensive business quality analysis"""
        
        # Format historical data
        historical_data = []
        for metric_name, trend_data in metrics_trends.items():
            if trend_data and len(trend_data) > 0:
                historical_data.append(f"{metric_name}: {trend_data}")
        
        years_context = ""
        if years:
            years_context = f"\nData covers years: {years}"
        
        return f"""
You are a seasoned business analyst specializing in quality assessment.

Company: {symbol}

Historical Business Performance:
{chr(10).join(historical_data)}{years_context}

ANALYSIS FRAMEWORK:
Evaluate the business quality across these dimensions:

1. CONSISTENCY: How stable are the key metrics over time?
2. GROWTH TRAJECTORY: Are metrics improving, declining, or stagnant?
3. COMPETITIVE POSITION: What do these trends suggest about market position?
4. MANAGEMENT EFFECTIVENESS: How well is management executing?
5. PREDICTABILITY: How predictable are future performance levels?

QUALITY RATING CRITERIA:
- HIGH: Consistent growth, stable margins, strong ROE trends, predictable performance
- MEDIUM: Some volatility but generally positive trends, acceptable consistency
- LOW: Declining trends, high volatility, deteriorating fundamentals

Provide analysis in JSON format:
{{
    "Business Quality Score": "[High/Medium/Low]",
    "Consistency Rating": "[Excellent/Good/Fair/Poor]",
    "Growth Trajectory": "[Strong Growth/Moderate Growth/Stable/Declining]",
    "Management Effectiveness": "[Excellent/Good/Average/Poor]",
    "Predictability": "[High/Medium/Low]",
    "AI Business Commentary": "[Detailed 3-4 sentence analysis explaining the rating]",
    "Key Strengths": "[List 2-3 main strengths]",
    "Areas of Concern": "[List any concerns or red flags]",
    "Future Outlook": "[Positive/Neutral/Concerning]"
}}

Base your analysis strictly on the provided historical data trends.
        """


class QuickBusinessAssessmentPrompt(BasePrompt):
    """
    Quick business quality assessment for rapid screening purposes.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'current_metrics']
    
    def generate(self, symbol: str, current_metrics: Dict[str, Any]) -> str:
        """Generate quick business assessment prompt"""
        
        metrics_summary = []
        for metric, value in current_metrics.items():
            metrics_summary.append(f"- {metric}: {value}")
        
        return f"""
Quick Business Quality Assessment for {symbol}

Current Financial Metrics:
{chr(10).join(metrics_summary)}

Based on these current metrics, provide a rapid business quality assessment:

Respond in JSON format:
{{
    "Quick Quality Score": "[High/Medium/Low]",
    "Primary Strength": "[Main positive aspect]",
    "Primary Concern": "[Main area of concern if any]",
    "One-Line Summary": "[Single sentence business quality summary]"
}}

Focus on the most critical indicators for business quality.
        """


class SectorComparativeQualityPrompt(BasePrompt):
    """
    Business quality prompt that includes sector/industry comparison context.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'company_metrics', 'sector_averages']
    
    def generate(self, symbol: str, company_metrics: Dict[str, Any], 
                sector_averages: Dict[str, Any], sector_name: str = "Industry") -> str:
        """Generate sector-comparative business quality analysis"""
        
        # Format company metrics
        company_data = []
        for metric, value in company_metrics.items():
            company_data.append(f"- {metric}: {value}")
        
        # Format sector averages
        sector_data = []
        for metric, value in sector_averages.items():
            sector_data.append(f"- {metric}: {value}")
        
        return f"""
Sector-Comparative Business Quality Analysis

Company: {symbol}
Sector: {sector_name}

COMPANY METRICS:
{chr(10).join(company_data)}

{sector_name.upper()} SECTOR AVERAGES:
{chr(10).join(sector_data)}

COMPARATIVE ANALYSIS FRAMEWORK:
Evaluate how {symbol} compares to its sector across:

1. RELATIVE PERFORMANCE: Above/below sector averages
2. COMPETITIVE POSITION: Market leadership indicators
3. SECTOR DYNAMICS: How company navigates industry challenges
4. RELATIVE VALUATION: Value vs sector peers

Provide comparative analysis in JSON format:
{{
    "Relative Quality Score": "[Superior/Above Average/Average/Below Average/Poor]",
    "Sector Ranking": "[Top Quartile/Second Quartile/Third Quartile/Bottom Quartile]",
    "Competitive Advantages": "[List key advantages vs peers]",
    "Relative Weaknesses": "[List areas where company lags sector]",
    "Sector Position Summary": "[2-3 sentence summary of competitive position]",
    "Investment Appeal vs Peers": "[Higher/Similar/Lower than sector average]"
}}

Focus on relative performance and competitive positioning within the sector.
        """
