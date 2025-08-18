"""
Analysis Prompts Library
=======================

Contains all prompts related to stock analysis, value investing evaluation,
and comprehensive financial assessment.
"""

from typing import Dict, Any
from abc import ABC, abstractmethod

class BasePrompt(ABC):
    """Base class for all prompts in the StockScreener system"""
    
    @abstractmethod
    def generate(self, **kwargs) -> str:
        """Generate the prompt string with provided parameters"""
        pass
    
    @property
    @abstractmethod
    def required_parameters(self) -> list:
        """Return list of required parameters for this prompt"""
        pass


class ValueInvestingAnalysisPrompt(BasePrompt):
    """
    Comprehensive value investing analysis prompt based on Graham & Buffett principles.
    Used for detailed financial analysis and investment recommendations.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'metrics', 'news']
    
    def generate(self, symbol: str, metrics: Dict[str, Any], news: str, ideal_ratios: Dict = None) -> str:
        """Generate comprehensive value investing analysis prompt"""
        
        # Create detailed ratio analysis section
        ratio_analysis = []
        if ideal_ratios:
            for metric, value in metrics.items():
                if metric in ideal_ratios and value is not None and value != 'N/A':
                    try:
                        numeric_value = float(value)
                        ideal_ranges = ideal_ratios[metric]
                        
                        # Determine if metric is in ideal range
                        category = 'poor'
                        for cat, (min_val, max_val) in ideal_ranges.items():
                            if min_val <= numeric_value <= max_val:
                                category = cat
                                break
                        
                        ratio_analysis.append(f"- {metric}: {numeric_value} ({category} range)")
                    except (ValueError, TypeError):
                        ratio_analysis.append(f"- {metric}: {value} (data issue)")
                else:
                    ratio_analysis.append(f"- {metric}: {value}")
        else:
            # Fallback for simple metric display
            for metric, value in metrics.items():
                ratio_analysis.append(f"- {metric}: {value}")

        return f"""
Conduct a comprehensive value investing analysis for this Indian stock based on REAL DATA:

COMPANY: {symbol}

FINANCIAL METRICS:
{chr(10).join(ratio_analysis)}

RECENT NEWS & MARKET DEVELOPMENTS:
{news if news else "Limited recent news available - analysis based primarily on financial metrics"}

CRITICAL INSTRUCTION: Base your analysis on REAL FACTS from the financial metrics and news provided above. Do NOT make assumptions about data not provided. If recent news indicates specific developments (earnings, partnerships, regulatory changes, management changes, etc.), factor these into your analysis.

VALUE INVESTING BENCHMARKS (Graham & Buffett Principles):
- PE Ratio: Excellent (8-15), Good (15-20), Acceptable (20-25), Poor (>25)
- Price to Book: Excellent (0.5-1.5), Good (1.5-2.5), Acceptable (2.5-3.5), Poor (>3.5)
- ROE: Excellent (20-50%), Good (15-20%), Acceptable (10-15%), Poor (<10%)
- Current Ratio: Excellent (2.0-3.0), Good (1.5-2.0), Acceptable (1.0-1.5), Poor (<1.0)
- Debt/Equity: Excellent (0-0.3), Good (0.3-0.6), Acceptable (0.6-1.0), Poor (>1.0)

Provide analysis covering:

**1. CURRENT SITUATION ASSESSMENT**
- Recent news impact on business fundamentals
- Financial metric analysis vs benchmarks
- Market sentiment from recent developments

**2. FINANCIAL HEALTH REALITY CHECK** 
- Actual liquidity position based on available data
- Real debt levels and their sustainability
- Profitability trends evident from metrics
- Warning signs or positive indicators

**3. BUSINESS QUALITY ANALYSIS**
- Competitive position based on news/developments
- Management effectiveness (if mentioned in news)
- Industry challenges/opportunities from recent news
- Revenue quality assessment

**4. RISK ASSESSMENT BASED ON FACTS**
- Immediate risks from recent developments
- Financial risks evident from metrics
- Market/industry risks from news context
- Regulatory or competitive threats mentioned

**5. VALUATION REALITY**
- Current valuation vs intrinsic value using available metrics
- Margin of safety calculation based on real data
- Recent market developments impact on fair value

**6. INVESTMENT DECISION**
- Clear recommendation based on facts provided
- Specific reasons backed by data/news
- Timeline and conditions for the recommendation

Format your response as:
VALUE_SCORE: [1-10]
FINANCIAL_HEALTH: [Excellent/Good/Fair/Poor]
BUSINESS_QUALITY: [High/Medium/Low]
RISK_LEVEL: [Low/Medium/High/Very High]
VALUATION: [Undervalued/Fairly Valued/Overvalued]
MARGIN_OF_SAFETY: [High/Medium/Low/None]
RECOMMENDATION: [Strong Buy/Buy/Hold/Sell/Avoid]
INVESTMENT_THESIS: [Factual 4-5 sentence analysis based on provided data and news]
KEY_RISKS: [Specific risks from actual data/news, not generic risks]
CATALYSTS: [Real upcoming events or developments mentioned in news]
TARGET_PRICE: [Fair value estimate based on available metrics]

IMPORTANT: 
1. Use EXACTLY the format above. Do NOT use markdown.
2. Base analysis ONLY on provided financial data and news.
3. If news mentions specific developments, incorporate them into reasoning.
4. Avoid generic statements - use specific facts from the data provided.
"""


class EnhancedAnalysisPrompt(BasePrompt):
    """
    Enhanced analysis prompt that includes price prediction context for more comprehensive analysis.
    Builds upon the base value investing prompt with additional prediction insights.
    """
    
    @property
    def required_parameters(self) -> list:
        return ['symbol', 'metrics', 'news', 'price_prediction']
    
    def generate(self, symbol: str, metrics: Dict[str, Any], news: str, 
                price_prediction: Dict, ideal_ratios: Dict = None) -> str:
        """Generate enhanced analysis prompt with price prediction context"""
        
        # Get the base prompt first
        base_prompt_generator = ValueInvestingAnalysisPrompt()
        basic_prompt = base_prompt_generator.generate(
            symbol=symbol, 
            metrics=metrics, 
            news=news, 
            ideal_ratios=ideal_ratios
        )
        
        # Add price prediction context if available
        if price_prediction and "error" not in price_prediction:
            predicted_price = price_prediction.get('predicted_price', 'N/A')
            confidence = price_prediction.get('confidence', 0)
            market_signal = price_prediction.get('market_signal', 'NEUTRAL')
            models_used = price_prediction.get('models_used', 0)
            
            # Get individual model predictions for context
            individual_predictions = price_prediction.get('individual_predictions', {})
            model_summary = []
            for model_name, pred_data in individual_predictions.items():
                if isinstance(pred_data, dict) and 'predicted_price' in pred_data:
                    model_summary.append(f"- {model_name}: ₹{pred_data['predicted_price']:.2f}")
            
            prediction_context = f"""

AI PRICE PREDICTION CONTEXT (ADDITIONAL INSIGHT):
- Ensemble Predicted Price: ₹{predicted_price}
- Confidence Level: {confidence:.1%}
- Market Signal: {market_signal}
- Models Consensus: {models_used}/6 models agree

Individual Model Predictions:
{chr(10).join(model_summary) if model_summary else "- No individual model data available"}

ENHANCED ANALYSIS INSTRUCTION: 
Incorporate the AI price prediction insights above into your valuation analysis. Consider:
1. How the predicted price compares with your fundamental analysis
2. Whether the AI confidence aligns with your risk assessment
3. If the market signal supports your recommendation
4. Any discrepancies between technical predictions and fundamental value

Your TARGET_PRICE should consider both fundamental analysis and AI prediction insights.
"""
            return basic_prompt + prediction_context
        
        return basic_prompt


# Template for custom analysis prompts
class CustomAnalysisPromptTemplate(BasePrompt):
    """
    Template for creating custom analysis prompts.
    Can be extended for specific analysis needs.
    """
    
    def __init__(self, template: str, required_params: list):
        self.template = template
        self._required_parameters = required_params
    
    @property
    def required_parameters(self) -> list:
        return self._required_parameters
    
    def generate(self, **kwargs) -> str:
        """Generate prompt using string formatting"""
        return self.template.format(**kwargs)


# Prompt validation utilities
def validate_prompt_parameters(prompt_class: BasePrompt, provided_params: Dict) -> bool:
    """Validate that all required parameters are provided for a prompt"""
    required = set(prompt_class.required_parameters)
    provided = set(provided_params.keys())
    missing = required - provided
    
    if missing:
        raise ValueError(f"Missing required parameters for {prompt_class.__class__.__name__}: {missing}")
    
    return True
