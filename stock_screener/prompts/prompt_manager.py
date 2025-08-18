"""
Prompt Manager
=============

Centralized management system for all StockScreener prompts.
Provides easy access, validation, and customization of prompts.
"""

from typing import Dict, Any, Type, Optional
import logging
from enum import Enum

from .analysis_prompts import (
    BasePrompt, ValueInvestingAnalysisPrompt, EnhancedAnalysisPrompt,
    CustomAnalysisPromptTemplate, validate_prompt_parameters
)
from .business_quality_prompts import (
    BusinessQualityAnalysisPrompt, ComprehensiveBusinessQualityPrompt,
    QuickBusinessAssessmentPrompt, SectorComparativeQualityPrompt
)
from .prediction_prompts import (
    PricePredictionPrompt, TechnicalAnalysisPrompt, FundamentalAnalysisPrompt,
    MultiPeriodPredictionPrompt, RiskAssessmentPrompt
)


class PromptType(Enum):
    """Enumeration of available prompt types"""
    VALUE_INVESTING_ANALYSIS = "value_investing_analysis"
    ENHANCED_ANALYSIS = "enhanced_analysis"
    BUSINESS_QUALITY = "business_quality"
    COMPREHENSIVE_BUSINESS_QUALITY = "comprehensive_business_quality"
    QUICK_BUSINESS_ASSESSMENT = "quick_business_assessment"
    SECTOR_COMPARATIVE_QUALITY = "sector_comparative_quality"
    PRICE_PREDICTION = "price_prediction"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MULTI_PERIOD_PREDICTION = "multi_period_prediction"
    RISK_ASSESSMENT = "risk_assessment"


class PromptManager:
    """
    Central manager for all StockScreener prompts.
    Provides easy access, validation, and generation of prompts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._prompt_registry = self._initialize_prompt_registry()
    
    def _initialize_prompt_registry(self) -> Dict[PromptType, Type[BasePrompt]]:
        """Initialize the registry of available prompts"""
        return {
            PromptType.VALUE_INVESTING_ANALYSIS: ValueInvestingAnalysisPrompt,
            PromptType.ENHANCED_ANALYSIS: EnhancedAnalysisPrompt,
            PromptType.BUSINESS_QUALITY: BusinessQualityAnalysisPrompt,
            PromptType.COMPREHENSIVE_BUSINESS_QUALITY: ComprehensiveBusinessQualityPrompt,
            PromptType.QUICK_BUSINESS_ASSESSMENT: QuickBusinessAssessmentPrompt,
            PromptType.SECTOR_COMPARATIVE_QUALITY: SectorComparativeQualityPrompt,
            PromptType.PRICE_PREDICTION: PricePredictionPrompt,
            PromptType.TECHNICAL_ANALYSIS: TechnicalAnalysisPrompt,
            PromptType.FUNDAMENTAL_ANALYSIS: FundamentalAnalysisPrompt,
            PromptType.MULTI_PERIOD_PREDICTION: MultiPeriodPredictionPrompt,
            PromptType.RISK_ASSESSMENT: RiskAssessmentPrompt,
        }
    
    def get_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """
        Generate a prompt of the specified type with provided parameters.
        
        Args:
            prompt_type: The type of prompt to generate
            **kwargs: Parameters required by the specific prompt
            
        Returns:
            Generated prompt string
            
        Raises:
            ValueError: If prompt type is not found or required parameters are missing
        """
        if prompt_type not in self._prompt_registry:
            available_types = [pt.value for pt in self._prompt_registry.keys()]
            raise ValueError(f"Prompt type '{prompt_type.value}' not found. Available: {available_types}")
        
        prompt_class = self._prompt_registry[prompt_type]
        prompt_instance = prompt_class()
        
        # Validate parameters
        try:
            validate_prompt_parameters(prompt_instance, kwargs)
        except ValueError as e:
            self.logger.error(f"Parameter validation failed for {prompt_type.value}: {e}")
            raise
        
        # Generate and return the prompt
        try:
            prompt_text = prompt_instance.generate(**kwargs)
            self.logger.info(f"Generated {prompt_type.value} prompt ({len(prompt_text)} characters)")
            return prompt_text
        except Exception as e:
            self.logger.error(f"Failed to generate {prompt_type.value} prompt: {e}")
            raise
    
    def get_required_parameters(self, prompt_type: PromptType) -> list:
        """Get the required parameters for a specific prompt type"""
        if prompt_type not in self._prompt_registry:
            raise ValueError(f"Prompt type '{prompt_type.value}' not found")
        
        prompt_class = self._prompt_registry[prompt_type]
        prompt_instance = prompt_class()
        return prompt_instance.required_parameters
    
    def list_available_prompts(self) -> Dict[str, list]:
        """List all available prompts and their required parameters"""
        available_prompts = {}
        for prompt_type, prompt_class in self._prompt_registry.items():
            prompt_instance = prompt_class()
            available_prompts[prompt_type.value] = {
                'class': prompt_class.__name__,
                'required_parameters': prompt_instance.required_parameters,
                'description': prompt_class.__doc__.strip() if prompt_class.__doc__ else "No description"
            }
        return available_prompts
    
    def register_custom_prompt(self, prompt_type: str, prompt_class: Type[BasePrompt]) -> None:
        """Register a custom prompt class"""
        custom_type = PromptType(prompt_type)  # This will create a new enum member
        self._prompt_registry[custom_type] = prompt_class
        self.logger.info(f"Registered custom prompt: {prompt_type}")
    
    def create_custom_template_prompt(self, name: str, template: str, required_params: list) -> None:
        """Create and register a custom template-based prompt"""
        custom_prompt = CustomAnalysisPromptTemplate(template, required_params)
        # Note: This would need enum extension for full implementation
        self.logger.info(f"Created custom template prompt: {name}")
        return custom_prompt


# Convenience functions for backward compatibility and ease of use
def get_value_investing_prompt(symbol: str, metrics: Dict[str, Any], news: str, 
                             ideal_ratios: Dict = None) -> str:
    """Convenience function for value investing analysis prompt"""
    manager = PromptManager()
    return manager.get_prompt(
        PromptType.VALUE_INVESTING_ANALYSIS,
        symbol=symbol,
        metrics=metrics,
        news=news,
        ideal_ratios=ideal_ratios
    )


def get_enhanced_analysis_prompt(symbol: str, metrics: Dict[str, Any], news: str,
                               price_prediction: Dict, ideal_ratios: Dict = None) -> str:
    """Convenience function for enhanced analysis prompt with price predictions"""
    manager = PromptManager()
    return manager.get_prompt(
        PromptType.ENHANCED_ANALYSIS,
        symbol=symbol,
        metrics=metrics,
        news=news,
        price_prediction=price_prediction,
        ideal_ratios=ideal_ratios
    )


def get_business_quality_prompt(symbol: str, roe_trend: list, margin_trend: list) -> str:
    """Convenience function for business quality analysis prompt"""
    manager = PromptManager()
    return manager.get_prompt(
        PromptType.BUSINESS_QUALITY,
        symbol=symbol,
        roe_trend=roe_trend,
        margin_trend=margin_trend
    )


# Global prompt manager instance
prompt_manager = PromptManager()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = PromptManager()
    
    # List available prompts
    print("Available Prompts:")
    for prompt_name, details in manager.list_available_prompts().items():
        print(f"- {prompt_name}: {details['required_parameters']}")
    
    # Example prompt generation
    try:
        sample_metrics = {
            'PE Ratio': 15.5,
            'ROE': 18.2,
            'Current Ratio': 2.1
        }
        
        prompt = manager.get_prompt(
            PromptType.VALUE_INVESTING_ANALYSIS,
            symbol='RELIANCE.NS',
            metrics=sample_metrics,
            news='Recent quarterly results show strong growth.'
        )
        print(f"\nGenerated prompt length: {len(prompt)} characters")
        
    except Exception as e:
        print(f"Error: {e}")
