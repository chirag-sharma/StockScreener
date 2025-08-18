# StockScreener Prompts Library Documentation

## Overview

The StockScreener Prompts Library is a centralized system for managing all AI prompts used throughout the application. This library provides organized, maintainable, and version-controlled prompts for various analysis tasks.

## Features

- **Structured Prompt Templates**: Object-oriented prompt design with clear inheritance
- **Parameter Validation**: Automatic validation of required parameters
- **Easy Customization**: Simple modification and extension of existing prompts
- **Version Control**: Track changes and updates to prompts over time
- **Fallback Support**: Graceful degradation to legacy prompts if needed
- **Type Safety**: Strong typing and clear interfaces

## Architecture

### Core Components

1. **BasePrompt**: Abstract base class for all prompts
2. **PromptManager**: Central manager for prompt generation and validation
3. **PromptType**: Enum defining all available prompt types
4. **Specialized Prompt Classes**: Domain-specific prompt implementations

### Directory Structure

```
stock_screener/prompts/
├── __init__.py                 # Main exports and version info
├── analysis_prompts.py         # Value investing and enhanced analysis prompts
├── business_quality_prompts.py # Business quality assessment prompts
├── prediction_prompts.py       # Price prediction and forecasting prompts
└── prompt_manager.py          # Central management and utilities
```

## Quick Start

### Basic Usage

```python
from stock_screener.prompts import PromptManager, PromptType

# Create prompt manager
manager = PromptManager()

# Generate a value investing analysis prompt
prompt = manager.get_prompt(
    PromptType.VALUE_INVESTING_ANALYSIS,
    symbol='RELIANCE.NS',
    metrics={'PE Ratio': 15.5, 'ROE': 18.2},
    news='Strong quarterly results'
)
```

### Convenience Functions

```python
from stock_screener.prompts import get_value_investing_prompt

# Use convenience function
prompt = get_value_investing_prompt(
    symbol='TCS.NS',
    metrics={'PE Ratio': 12.3, 'ROE': 22.1},
    news='Record client wins in Q4'
)
```

## Available Prompt Types

### Analysis Prompts

#### VALUE_INVESTING_ANALYSIS
Comprehensive value investing analysis based on Graham & Buffett principles.

**Required Parameters:**
- `symbol`: Stock symbol (e.g., 'RELIANCE.NS')
- `metrics`: Dictionary of financial metrics
- `news`: Recent news and developments
- `ideal_ratios`: (Optional) Benchmark ratios for comparison

**Example:**
```python
prompt = manager.get_prompt(
    PromptType.VALUE_INVESTING_ANALYSIS,
    symbol='HDFC.NS',
    metrics={
        'PE Ratio': 18.5,
        'ROE': 16.2,
        'Current Ratio': 1.8,
        'Debt/Equity': 0.6
    },
    news='Banking sector showing strong recovery'
)
```

#### ENHANCED_ANALYSIS
Extended analysis that includes AI price prediction context.

**Required Parameters:**
- `symbol`: Stock symbol
- `metrics`: Financial metrics dictionary
- `news`: Recent news
- `price_prediction`: Price prediction results from AI models

### Business Quality Prompts

#### BUSINESS_QUALITY
Basic business quality assessment based on historical trends.

**Required Parameters:**
- `symbol`: Stock symbol
- `roe_trend`: List of ROE values over time
- `margin_trend`: List of operating margin values

#### COMPREHENSIVE_BUSINESS_QUALITY
Enhanced business quality analysis with multiple metrics.

**Required Parameters:**
- `symbol`: Stock symbol
- `metrics_trends`: Dictionary of metric trends over time

### Prediction Prompts

#### PRICE_PREDICTION
Analysis and interpretation of AI price predictions.

#### TECHNICAL_ANALYSIS
Technical analysis prompt for chart patterns and indicators.

#### MULTI_PERIOD_PREDICTION
Analysis of predictions across multiple time horizons.

## Creating Custom Prompts

### Method 1: Extending BasePrompt

```python
from stock_screener.prompts.analysis_prompts import BasePrompt

class MyCustomPrompt(BasePrompt):
    @property
    def required_parameters(self) -> list:
        return ['param1', 'param2']
    
    def generate(self, param1: str, param2: int) -> str:
        return f"Custom prompt for {param1} with value {param2}"
```

### Method 2: Using Template

```python
from stock_screener.prompts import CustomAnalysisPromptTemplate

template = """
Analyze {company} with the following data:
- Metric: {metric_value}
- Context: {context}
"""

custom_prompt = CustomAnalysisPromptTemplate(
    template=template,
    required_params=['company', 'metric_value', 'context']
)
```

## Integration with Existing Code

The prompts library is designed for seamless integration with existing StockScreener components:

### Analyzer Integration

```python
# In analyzer.py
from ..prompts import PromptManager, PromptType

def _create_ai_analysis_prompt(self, symbol, metrics, news):
    if HAS_PROMPTS_LIBRARY:
        prompt_manager = PromptManager()
        return prompt_manager.get_prompt(
            PromptType.VALUE_INVESTING_ANALYSIS,
            symbol=symbol,
            metrics=metrics,
            news=news,
            ideal_ratios=self.ideal_ratios
        )
    # Fallback to legacy prompt...
```

### Service Integration

```python
# In business quality service
from ..prompts import get_business_quality_prompt

def analyze_business_quality(symbol, roe_trend, margin_trend):
    prompt = get_business_quality_prompt(
        symbol=symbol,
        roe_trend=roe_trend,
        margin_trend=margin_trend
    )
    # Send to AI API...
```

## Best Practices

### 1. Parameter Validation
Always validate parameters before generating prompts:

```python
try:
    prompt = manager.get_prompt(PromptType.VALUE_INVESTING_ANALYSIS, **params)
except ValueError as e:
    logger.error(f"Invalid parameters: {e}")
```

### 2. Error Handling
Implement graceful fallbacks:

```python
try:
    prompt = manager.get_prompt(prompt_type, **params)
except Exception as e:
    logger.warning(f"Prompts library failed: {e}, using legacy")
    prompt = legacy_prompt_function(**params)
```

### 3. Consistent Formatting
Use the structured output formats defined in prompts:

```python
# Prompts are designed to return structured responses
# VALUE_SCORE: [1-10]
# FINANCIAL_HEALTH: [Excellent/Good/Fair/Poor]
# RECOMMENDATION: [Strong Buy/Buy/Hold/Sell/Avoid]
```

## Advanced Features

### Prompt Versioning

Track prompt changes over time by updating version numbers:

```python
class ValueInvestingAnalysisPromptV2(ValueInvestingAnalysisPrompt):
    """Updated prompt with enhanced risk assessment"""
    
    def generate(self, **kwargs) -> str:
        # Enhanced prompt logic
        pass
```

### Dynamic Parameter Injection

Add contextual information dynamically:

```python
def generate_contextual_prompt(base_params, market_context):
    if market_context == 'bear_market':
        base_params['additional_context'] = 'Consider defensive positioning'
    
    return manager.get_prompt(PromptType.VALUE_INVESTING_ANALYSIS, **base_params)
```

### Batch Processing

Generate multiple prompts efficiently:

```python
stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFC.NS']
prompts = []

for stock in stocks:
    prompt = manager.get_prompt(
        PromptType.VALUE_INVESTING_ANALYSIS,
        symbol=stock,
        metrics=get_metrics(stock),
        news=get_news(stock)
    )
    prompts.append(prompt)
```

## Migration Guide

### From Legacy to New Library

1. **Replace direct string formatting:**
   ```python
   # Old way
   prompt = f"Analyze {symbol} with metrics {metrics}"
   
   # New way
   prompt = manager.get_prompt(PromptType.VALUE_INVESTING_ANALYSIS, 
                              symbol=symbol, metrics=metrics, news=news)
   ```

2. **Update imports:**
   ```python
   # Add to existing imports
   from stock_screener.prompts import PromptManager, PromptType
   ```

3. **Add error handling:**
   ```python
   try:
       prompt = manager.get_prompt(...)
   except ImportError:
       # Library not available, use legacy
   except ValueError:
       # Invalid parameters
   ```

## Troubleshooting

### Common Issues

1. **ImportError**: Prompts library not found
   - Ensure `stock_screener/prompts/` directory exists
   - Check that `__init__.py` is present

2. **ValueError**: Missing required parameters
   - Check `get_required_parameters()` for the prompt type
   - Ensure all required fields are provided

3. **Template errors**: String formatting issues
   - Validate parameter types match template expectations
   - Check for special characters in parameter values

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('stock_screener.prompts').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Multi-language Support**: Prompts in different languages
- **A/B Testing**: Compare prompt effectiveness
- **Dynamic Optimization**: AI-powered prompt improvement
- **Template Library**: Pre-built templates for common scenarios
- **Integration Testing**: Automated validation of prompt outputs

## Contributing

When adding new prompts:

1. Extend `BasePrompt` class
2. Define clear `required_parameters`
3. Implement robust `generate()` method
4. Add to `PromptType` enum
5. Register in `PromptManager`
6. Add documentation and examples
7. Create unit tests

---

*This documentation is part of the StockScreener v3.0 professional upgrade.*
