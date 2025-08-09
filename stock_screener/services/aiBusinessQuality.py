"""
This module provides AI-powered business quality analysis using OpenAI's GPT models.
It sends financial trends to the model and parses the response for business quality assessment.
"""

import openai
import logging
import os
import json

# Setup logging for the module
logging.basicConfig(level=logging.INFO)

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    # Construct the prompt for the AI model
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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # or gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = response['choices'][0]['message']['content']
        logging.info(f"[AI] Business Quality Response: {content}")
        # Use json.loads for safe parsing instead of eval
        return json.loads(content)
    except Exception as e:
        logging.error(f"[AI] Error during business quality check: {e}")
        return {
            "Business Quality Score": "N/A",
            "AI Business Commentary": "Unable to evaluate due to error or insufficient data."
        }
