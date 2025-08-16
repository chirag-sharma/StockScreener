"""
StockScreener Prompts Library
============================

A centralized library for managing all AI prompts used in the StockScreener application.
This module provides organized, maintainable prompts for various analysis tasks.

Features:
- Structured prompt templates
- Easy customization and updates
- Version control for prompt changes
- Consistent formatting across the application
"""

# Import main classes and functions
from .analysis_prompts import (
    BasePrompt,
    ValueInvestingAnalysisPrompt,
    EnhancedAnalysisPrompt,
    CustomAnalysisPromptTemplate,
    validate_prompt_parameters
)

from .business_quality_prompts import (
    BusinessQualityAnalysisPrompt,
    ComprehensiveBusinessQualityPrompt,
    QuickBusinessAssessmentPrompt,
    SectorComparativeQualityPrompt
)

from .prediction_prompts import (
    PricePredictionPrompt,
    TechnicalAnalysisPrompt,
    FundamentalAnalysisPrompt,
    MultiPeriodPredictionPrompt,
    RiskAssessmentPrompt
)

from .prompt_manager import (
    PromptManager,
    PromptType,
    get_value_investing_prompt,
    get_enhanced_analysis_prompt,
    get_business_quality_prompt,
    prompt_manager
)

__version__ = "1.0.0"
__all__ = [
    # Base classes
    'BasePrompt',
    'validate_prompt_parameters',
    
    # Analysis prompts
    'ValueInvestingAnalysisPrompt',
    'EnhancedAnalysisPrompt',
    'CustomAnalysisPromptTemplate',
    
    # Business quality prompts
    'BusinessQualityAnalysisPrompt',
    'ComprehensiveBusinessQualityPrompt',
    'QuickBusinessAssessmentPrompt',
    'SectorComparativeQualityPrompt',
    
    # Prediction prompts
    'PricePredictionPrompt',
    'TechnicalAnalysisPrompt',
    'FundamentalAnalysisPrompt',
    'MultiPeriodPredictionPrompt',
    'RiskAssessmentPrompt',
    
    # Management
    'PromptManager',
    'PromptType',
    'prompt_manager',
    
    # Convenience functions
    'get_value_investing_prompt',
    'get_enhanced_analysis_prompt',
    'get_business_quality_prompt',
]
