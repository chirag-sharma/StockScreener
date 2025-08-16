#!/usr/bin/env python3
"""
StockScreener Prompts Library - Example Usage
=============================================

This script demonstrates how to use the StockScreener Prompts Library
for various analysis tasks.
"""

from stock_screener.prompts import (
    PromptManager, 
    PromptType, 
    get_value_investing_prompt,
    get_business_quality_prompt
)

def main():
    print("🚀 StockScreener Prompts Library Examples")
    print("=" * 50)
    
    # Initialize the prompt manager
    manager = PromptManager()
    
    # Example 1: Value Investing Analysis
    print("\n📊 Example 1: Value Investing Analysis")
    print("-" * 40)
    
    sample_metrics = {
        'PE Ratio': 15.8,
        'ROE': 18.5,
        'Current Ratio': 2.1,
        'Debt/Equity': 0.45,
        'Price to Book': 2.8,
        'Operating Margin': 12.3
    }
    
    sample_news = """
    Reliance Industries reported strong Q4 results with 15% YoY revenue growth. 
    The company announced new investments in renewable energy sector worth ₹75,000 crores. 
    Management guided for continued expansion in retail and telecom segments.
    """
    
    try:
        value_prompt = manager.get_prompt(
            PromptType.VALUE_INVESTING_ANALYSIS,
            symbol='RELIANCE.NS',
            metrics=sample_metrics,
            news=sample_news
        )
        
        print(f"✅ Generated value investing prompt: {len(value_prompt)} characters")
        print(f"Preview: {value_prompt[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Business Quality Analysis
    print("\n🏢 Example 2: Business Quality Analysis")
    print("-" * 40)
    
    roe_trends = [16.2, 17.8, 18.5, 19.1, 18.9]  # Last 5 years
    margin_trends = [11.5, 12.1, 12.8, 12.3, 12.9]  # Operating margins
    
    try:
        business_prompt = get_business_quality_prompt(
            symbol='TCS.NS',
            roe_trend=roe_trends,
            margin_trend=margin_trends
        )
        
        print(f"✅ Generated business quality prompt: {len(business_prompt)} characters")
        print(f"Preview: {business_prompt[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Enhanced Analysis with Price Prediction
    print("\n🔮 Example 3: Enhanced Analysis with AI Predictions")
    print("-" * 50)
    
    mock_prediction = {
        'predicted_price': 2875.50,
        'confidence': 0.82,
        'market_signal': 'BUY',
        'models_used': 5,
        'individual_predictions': {
            'technical': {'predicted_price': 2880.0},
            'fundamental': {'predicted_price': 2900.0},
            'machine_learning': {'predicted_price': 2850.0},
            'time_series': {'predicted_price': 2870.0},
            'pattern_recognition': {'predicted_price': 2875.0}
        }
    }
    
    try:
        enhanced_prompt = manager.get_prompt(
            PromptType.ENHANCED_ANALYSIS,
            symbol='TCS.NS',
            metrics=sample_metrics,
            news="TCS announces major deal with Fortune 500 client",
            price_prediction=mock_prediction
        )
        
        print(f"✅ Generated enhanced analysis prompt: {len(enhanced_prompt)} characters")
        print(f"Includes AI prediction context: {mock_prediction['predicted_price']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: List Available Prompts
    print("\n📋 Example 4: Available Prompt Types")
    print("-" * 40)
    
    available_prompts = manager.list_available_prompts()
    for prompt_name, details in available_prompts.items():
        print(f"• {prompt_name}")
        print(f"  Class: {details['class']}")
        print(f"  Required: {details['required_parameters']}")
        print()
    
    # Example 5: Parameter Validation
    print("🔍 Example 5: Parameter Validation")
    print("-" * 40)
    
    try:
        # This should fail due to missing parameters
        manager.get_prompt(PromptType.VALUE_INVESTING_ANALYSIS, symbol='TEST.NS')
    except ValueError as e:
        print(f"✅ Validation working: {e}")
    
    # Example 6: Custom Template Usage
    print("\n🛠️ Example 6: Custom Template")
    print("-" * 30)
    
    from stock_screener.prompts.analysis_prompts import CustomAnalysisPromptTemplate
    
    custom_template = """
Sector Analysis for {sector}

Key Metrics:
- Leader: {market_leader}
- Growth Rate: {growth_rate}%
- Market Size: {market_size}

Provide analysis of sector trends and investment opportunities.
    """
    
    custom_prompt = CustomAnalysisPromptTemplate(
        template=custom_template,
        required_params=['sector', 'market_leader', 'growth_rate', 'market_size']
    )
    
    try:
        sector_analysis = custom_prompt.generate(
            sector="Information Technology",
            market_leader="TCS",
            growth_rate=12.5,
            market_size="$200B"
        )
        
        print(f"✅ Custom template prompt generated: {len(sector_analysis)} characters")
        print(f"Content:\n{sector_analysis}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 All examples completed successfully!")
    print("\nNext Steps:")
    print("• Integrate prompts into your analysis workflow")
    print("• Create custom prompts for specific needs")
    print("• Use PromptManager for centralized prompt management")
    

if __name__ == "__main__":
    main()
