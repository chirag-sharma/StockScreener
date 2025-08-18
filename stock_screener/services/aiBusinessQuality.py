"""
This module provides AI-powered business quality analysis using OpenAI's GPT models.
It sends financial trends to the model and parses the response for business quality assessment.
"""

import openai
import logging
import os
import json
import time

# Import prompts library
try:
    from ..prompts import get_business_quality_prompt, PromptManager, PromptType
    HAS_PROMPTS_LIBRARY = True
except ImportError:
    HAS_PROMPTS_LIBRARY = False
    print("Warning: Prompts library not available, using legacy prompts")
from datetime import datetime

# Setup logging for the module
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client with the new API format
def get_openai_client():
    """Get OpenAI client with proper API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return openai.OpenAI(api_key=api_key)
    return None

def get_business_quality_analysis(symbol, roe_trend, margin_trend):
    """
    Send ROE and margin trends to GPT and get business quality assessment.

    Args:
        symbol (str): Stock symbol.
        roe_trend (list or str): Historical ROE values or trend description.
        margin_trend (list or str): Historical operating margin values or trend description.

    Returns:
        dict: Contains 'Business Quality Score' and 'AI Business Commentary'.
    """
    # Log AI analysis initiation
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🤖 [AI ANALYSIS START] {timestamp}")
    print(f"📊 Analyzing business quality for {symbol}")
    print(f"   ROE Trend: {roe_trend}")
    print(f"   Margin Trend: {margin_trend}")
    
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        print(f"❌ [AI API ERROR] OpenAI API key not available")
        return {
            "Business Quality Score": "N/A",
            "AI Business Commentary": "OpenAI API key not configured"
        }
    
    # Construct the prompt for the AI model
    if HAS_PROMPTS_LIBRARY:
        # Use the new prompts library
        try:
            prompt = get_business_quality_prompt(
                symbol=symbol,
                roe_trend=roe_trend,
                margin_trend=margin_trend
            )
        except Exception as e:
            print(f"Warning: Failed to use prompts library ({e}), falling back to legacy prompt")
            # Fallback to legacy prompt
            prompt = f"""
You are a value investing analyst.

A stock {symbol} has the following historical performance:

ROE (%): {roe_trend}
Operating Margin (%): {margin_trend}

Analyze the business quality. Is it consistent, improving, or declining? Give a short analysis and rate the business quality as one of: High, Medium, Low.
Respond in JSON format with fields:
- "Business Quality Score"
- "AI Business Commentary"
    """
    else:
        # Legacy prompt
        prompt = f"""
You are a value investing analyst.

A stock {symbol} has the following historical performance:

ROE (%): {roe_trend}
Operating Margin (%): {margin_trend}

Analyze the business quality. Is it consistent, improving, or declining? Give a short analysis and rate the business quality as one of: High, Medium, Low.
Respond in JSON format with fields:
- "Business Quality Score"
- "AI Business Commentary"
    """

    print(f"🔍 [AI PROMPT] Sending prompt to OpenAI GPT-4...")
    print(f"   Prompt length: {len(prompt)} characters")
    
    try:
        # Track API call timing
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        # Calculate API response time
        api_duration = time.time() - api_start_time
        
        # Extract response content
        content = response.choices[0].message.content
        
        # Log comprehensive API response details
        print(f"✅ [AI API SUCCESS] Response received in {api_duration:.2f}s")
        print(f"   Model: {response.model}")
        print(f"   Usage - Prompt tokens: {response.usage.prompt_tokens}")
        print(f"   Usage - Completion tokens: {response.usage.completion_tokens}")
        print(f"   Usage - Total tokens: {response.usage.total_tokens}")
        print(f"   Response length: {len(content)} characters")
        print(f"   Raw response: {content[:200]}{'...' if len(content) > 200 else ''}")
        
        # Log the JSON parsing attempt
        logging.info(f"[AI] Business Quality Raw Response for {symbol}: {content}")
        
        try:
            # Clean content - remove markdown code block markers if present
            clean_content = content.strip()
            if clean_content.startswith('```json'):
                clean_content = clean_content.replace('```json', '').strip()
            if clean_content.endswith('```'):
                clean_content = clean_content.replace('```', '').strip()
            
            # Use json.loads for safe parsing instead of eval
            parsed_response = json.loads(clean_content)
            print(f"✅ [JSON PARSING] Successfully parsed AI response")
            print(f"   Business Quality Score: {parsed_response.get('Business Quality Score', 'N/A')}")
            print(f"   AI Commentary: {parsed_response.get('AI Business Commentary', 'N/A')[:100]}{'...' if len(str(parsed_response.get('AI Business Commentary', ''))) > 100 else ''}")
            
            return parsed_response
            
        except json.JSONDecodeError as json_error:
            print(f"❌ [JSON PARSING ERROR] Failed to parse AI response as JSON: {json_error}")
            print(f"   Raw content: {content}")
            # Fallback response structure
            return {
                "Business Quality Score": "N/A",
                "AI Business Commentary": f"Parsing error: {content[:200]}"
            }
            
    except Exception as e:
        print(f"❌ [AI API ERROR] OpenAI API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        logging.error(f"[AI] Error during business quality check for {symbol}: {e}")
        return {
            "Business Quality Score": "N/A",
            "AI Business Commentary": "Unable to evaluate due to error or insufficient data."
        }
