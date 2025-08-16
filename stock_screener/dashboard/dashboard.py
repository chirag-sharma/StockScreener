"""
Professional AI-Powered Stock Screener Dashboard v3.0 - Working Version
======================================================================

A comprehensive Streamlit dashboard featuring:
- Professional UI with modern styling  
- Compatible with actual Excel data structure
- Robust error handling
- Enhanced visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import yfinance as yf
import time

# Try to import the price prediction service with error handling
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Use new modular prediction system
    from stock_screener.prediction_models import PricePredictionOrchestrator, predict_stock_price
    HAS_PRICE_PREDICTION = True
except ImportError:
    HAS_PRICE_PREDICTION = False
    print("Price Prediction Service not available - using basic price charts only")

# Enhanced page configuration with dark theme
st.set_page_config(
    page_title="AI Stock Screener Pro",
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Dark Theme CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Dark Theme Variables */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --bg-quaternary: #30363d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --border-primary: #30363d;
        --border-secondary: #21262d;
        --accent-primary: #58a6ff;
        --accent-secondary: #7c3aed;
        --success: #3fb950;
        --warning: #d29922;
        --danger: #f85149;
        --info: #79c0ff;
    }
    
    /* Main Container Dark Theme */
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Dark Theme */
    .css-1d391kg, .css-1lcbmhc, .stSidebar > div {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-primary) !important;
    }
    
    .css-1lcbmhc .stSelectbox label, .css-1lcbmhc .stMultiSelect label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Main Content Area */
    .main .block-container {
        background: var(--bg-primary) !important;
        padding-top: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Headers with Modern Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.02em;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid var(--border-primary);
        position: relative;
    }
    
    .sub-header::before {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    }
    
    /* Modern Card Styling */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-primary);
        box-shadow: 0 20px 40px rgba(88, 166, 255, 0.1);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    /* Status Badges with Modern Design */
    .grade-strong-buy {
        background: linear-gradient(135deg, var(--success), #2da44e);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(63, 185, 80, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .grade-buy {
        background: linear-gradient(135deg, var(--accent-primary), #4493f8);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .grade-hold {
        background: linear-gradient(135deg, var(--warning), #bf8700);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(210, 153, 34, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .grade-avoid {
        background: linear-gradient(135deg, var(--danger), #da3633);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(248, 81, 73, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Modern Status Indicators */
    .status-online {
        color: var(--success);
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-online::before {
        content: '●';
        font-size: 0.75rem;
        animation: pulse 2s infinite;
    }
    
    .performance-positive {
        color: var(--success);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .performance-negative {
        color: var(--danger);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Enhanced Price Cards */
    .price-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .price-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .price-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-primary);
        box-shadow: 0 12px 32px rgba(88, 166, 255, 0.15);
    }
    
    .price-card:hover::before {
        opacity: 1;
    }
    
    .price-positive {
        color: var(--success);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .price-negative {
        color: var(--danger);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Modern Recommendation Badges */
    .recommendation-strong-buy, .recommendation-buy, .recommendation-hold {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .recommendation-strong-buy {
        background: var(--success);
        color: white;
    }
    
    .recommendation-buy {
        background: var(--accent-primary);
        color: white;
    }
    
    .recommendation-hold {
        background: var(--warning);
        color: white;
    }
    
    /* Enhanced Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
    }
    
    .metrics-grid-item {
        text-align: center;
        padding: 1rem;
        background: var(--bg-tertiary);
        border-radius: 12px;
        border: 1px solid var(--border-secondary);
        transition: all 0.3s ease;
    }
    
    .metrics-grid-item:hover {
        transform: translateY(-2px);
        border-color: var(--accent-primary);
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.1);
    }
    
    .metrics-label {
        font-weight: 500;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metrics-value {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.5rem;
    }
    
    .metrics-score {
        color: var(--accent-primary);
    }
    
    .metrics-current {
        color: var(--text-primary);
    }
    
    .metrics-target {
        color: var(--success);
    }
    
    .metrics-upside {
        color: var(--success);
    }
    
    .metrics-negative {
        color: var(--danger);
    }
    
    /* Streamlit Component Overrides */
    .stSelectbox > div > div > div {
        background: var(--bg-tertiary) !important;
        border-color: var(--border-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .stMultiSelect > div > div > div {
        background: var(--bg-tertiary) !important;
        border-color: var(--border-primary) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        border: 1px solid var(--border-primary) !important;
        gap: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-primary) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3) !important;
    }
    
    .stDataFrame {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    .stDataFrame table {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stDataFrame th {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--border-primary) !important;
        padding: 1rem 0.75rem !important;
    }
    
    .stDataFrame td {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-secondary) !important;
        padding: 0.875rem 0.75rem !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.3) !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success), #2da44e) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(63, 185, 80, 0.3) !important;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
    }
    
    .stSlider .stSlider-thumb {
        background: var(--accent-primary) !important;
    }
    
    .stSlider .stSlider-track {
        background: var(--border-primary) !important;
    }
    
    .stSlider .stSlider-trackActive {
        background: var(--accent-primary) !important;
    }
    
    /* Plotly Chart Styling */
    .js-plotly-plot {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 40px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def format_currency(value):
    """Format currency values in readable format"""
    if pd.isna(value) or value is None or str(value).lower() == 'n/a':
        return "N/A"
    try:
        # Handle string 'N/A' or other non-numeric strings
        if isinstance(value, str) and not value.replace('.', '').replace(',', '').replace('-', '').isdigit():
            return "N/A"
        
        value = float(value)
        
        # Show N/A for zero or negative values (likely data issues)
        if value <= 0:
            return "N/A"
            
        if value >= 10000:
            return f"₹{value:,.0f}"
        else:
            return f"₹{value:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value):
    """Format percentage values"""
    if pd.isna(value):
        return "N/A"
    try:
        value = float(value)
        if value > 0:
            return f"+{value:.2f}%"
        else:
            return f"{value:.2f}%"
    except:
        return "N/A"

def get_current_price(symbol):
    """Fetch current price from yfinance"""
    try:
        # Check if symbol already has .NS suffix
        if not symbol.endswith('.NS'):
            ticker_symbol = f"{symbol}.NS"
        else:
            ticker_symbol = symbol
            
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = ticker.info.get('previousClose', hist['Close'].iloc[-1])
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            return current_price, change, change_pct
        return None, None, None
    except:
        return None, None, None

def get_recommendation_style(recommendation):
    """Get CSS class for recommendation styling"""
    if pd.isna(recommendation):
        return ""
    rec_lower = str(recommendation).lower()
    if "strong buy" in rec_lower:
        return "recommendation-strong-buy"
    elif "buy" in rec_lower:
        return "recommendation-buy"
    elif "hold" in rec_lower:
        return "recommendation-hold"
    return ""

def load_data():
    """
    Load and process the Excel data with priority system:
    
    Priority 1: comprehensive_analysis.xlsx (latest AI analysis)
    Priority 2: comprehensive_analysis_*.xlsx (datetime-stamped fallback)
    
    Returns:
        tuple: (DataFrame or None, status_message)
    """
    try:
        output_dir = Path("data/output")
        if not output_dir.exists():
            return None, "Output directory not found"
        
        # First, try to load the latest comprehensive analysis file (from AI screener)
        latest_file = output_dir / "comprehensive_analysis.xlsx"
        if latest_file.exists():
            # Use the latest comprehensive analysis file (Priority 1)
            pass
        else:
            # Fallback: Find Excel files with datetime stamps (Priority 2)
            analysis_files = list(output_dir.glob("comprehensive_analysis_*.xlsx"))
            if not analysis_files:
                return None, "No analysis files found"
            
            # Get latest file by modification time
            latest_file = max(analysis_files, key=os.path.getmtime)
        
        # Load data
        df = pd.read_excel(latest_file)
        
        if df.empty:
            return None, "Data file is empty"
        
        # Standardize column names for easier access
        column_mapping = {
            'Symbol': 'symbol',
            'Company Name': 'company_name',
            'Current Price (₹)': 'current_price',
            'Market Cap (Cr)': 'market_cap',  # Fixed: actual column name with (Cr)
            'Value Score': 'value_score',
            'Investment Recommendation': 'investment_grade',
            'Price Target (12M)': 'target_price',  # Fixed: use the numeric column
            'Business Quality': 'business_quality',
            'AI Reasoning': 'ai_reasoning',
            'AI Sentiment': 'ai_sentiment',
            'Financial Health': 'financial_health',
            'PE Ratio': 'pe_ratio',
            'ROE': 'roe',
            'Debt/Equity': 'debt_equity'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Data processing and formatting fixes
        try:
            # Fix Market Cap - it's already in Crores in 'Market Cap (Cr)' column
            if 'market_cap' in df.columns:
                df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
                df['market_cap'] = df['market_cap'].fillna(0)
            
            # Ensure target_price is numeric and handle NaN values
            if 'target_price' in df.columns:
                # Clean currency symbols and convert to numeric
                if df['target_price'].dtype == 'object':
                    # Remove currency symbols like ₹, $, etc. and convert to float
                    df['target_price'] = df['target_price'].astype(str).str.replace(r'[₹$€£¥,]', '', regex=True)
                df['target_price'] = pd.to_numeric(df['target_price'], errors='coerce')
                df['target_price'] = df['target_price'].fillna(0)
            
            # Get current prices if not available in data
            if 'current_price' not in df.columns and 'symbol' in df.columns:
                st.info("Fetching current prices from market data...")
                current_prices = []
                
                for symbol in df['symbol']:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period='1d')
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            current_prices.append(current_price)
                        else:
                            current_prices.append(0)
                    except:
                        current_prices.append(0)
                
                df['current_price'] = current_prices
            elif 'current_price' in df.columns:
                df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0)
            
            # Fill any remaining NaN values in numeric columns
            numeric_columns = ['value_score', 'pe_ratio', 'roe', 'debt_equity']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
        except Exception as e:
            st.warning(f"Warning: Data processing issue - {str(e)}")
        
        return df, f"Loaded {len(df)} stocks from {latest_file.name}"
        
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def show_status_bar():
    """Display system status"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.html('<span class="status-online">🟢 AI System: Online</span>')
    with col2:
        st.html('<span class="status-online">🟢 Data Feed: Live</span>')
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S IST")
        st.html(f'<span class="status-online">🟢 Last Update: {current_time}</span>')

def create_modern_summary_metrics(df):
    """Create modern summary metrics cards with dark theme"""
    st.html('<div class="sub-header">📊 Portfolio Overview</div>')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(df)
        metric_html = f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                ">📈</div>
                <div>
                    <div style="
                        font-size: 0.875rem;
                        color: var(--text-secondary);
                        font-weight: 500;
                        margin-bottom: 0.25rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Total Stocks</div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: var(--text-primary);
                        font-family: 'JetBrains Mono', monospace;
                    ">{total_stocks}</div>
                </div>
            </div>
        </div>
        """
        st.html(metric_html)
    
    with col2:
        if 'investment_grade' in df.columns:
            strong_buys = len(df[df['investment_grade'].str.contains('Strong Buy|STRONG_BUY', case=False, na=False)])
        else:
            strong_buys = 0
        metric_html = f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, var(--success), #2da44e);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                ">🎯</div>
                <div>
                    <div style="
                        font-size: 0.875rem;
                        color: var(--text-secondary);
                        font-weight: 500;
                        margin-bottom: 0.25rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Strong Buys</div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: var(--success);
                        font-family: 'JetBrains Mono', monospace;
                    ">{strong_buys}</div>
                </div>
            </div>
        </div>
        """
        st.html(metric_html)
    
    with col3:
        if 'value_score' in df.columns:
            avg_score = df['value_score'].mean()
        else:
            avg_score = 0
        metric_html = f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, var(--warning), #bf8700);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                ">⭐</div>
                <div>
                    <div style="
                        font-size: 0.875rem;
                        color: var(--text-secondary);
                        font-weight: 500;
                        margin-bottom: 0.25rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Avg Score</div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: var(--warning);
                        font-family: 'JetBrains Mono', monospace;
                    ">{avg_score:.1f}</div>
                </div>
            </div>
        </div>
        """
        st.html(metric_html)
    
    with col4:
        if 'target_price' in df.columns and 'current_price' in df.columns:
            # Handle potential division by zero and NaN values
            valid_data = df.dropna(subset=['target_price', 'current_price'])
            valid_data = valid_data[valid_data['current_price'] > 0]
            if len(valid_data) > 0:
                avg_upside = ((valid_data['target_price'] - valid_data['current_price']) / valid_data['current_price'] * 100).mean()
            else:
                avg_upside = 0
        else:
            avg_upside = 0
        
        # Proper conditional styling for upside values
        if avg_upside > 0:
            upside_color = "var(--success)"
            upside_icon = "📈"
        elif avg_upside < 0:
            upside_color = "var(--danger)"
            upside_icon = "�"
        else:  # avg_upside == 0 or N/A
            upside_color = "var(--text-secondary)"
            upside_icon = "�"
        
        metric_html = f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, {upside_color}, {upside_color}dd);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                ">{upside_icon}</div>
                <div>
                    <div style="
                        font-size: 0.875rem;
                        color: var(--text-secondary);
                        font-weight: 500;
                        margin-bottom: 0.25rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Avg Upside</div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: {upside_color};
                        font-family: 'JetBrains Mono', monospace;
                    ">{avg_upside:+.1f}%</div>
                </div>
            </div>
        </div>
        """
        st.html(metric_html)

def get_dark_theme_config():
    """Get dark theme configuration for Plotly charts"""
    return {
        'layout': {
            'paper_bgcolor': '#161b22',
            'plot_bgcolor': '#0d1117',
            'font': {
                'color': '#f0f6fc',
                'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
            },
            'colorway': ['#58a6ff', '#7c3aed', '#3fb950', '#d29922', '#f85149', '#79c0ff', '#bc8cff', '#56d4dd'],
            'title': {
                'font': {'color': '#f0f6fc', 'size': 18, 'family': 'Inter'},
                'x': 0.5
            },
            'xaxis': {
                'gridcolor': '#30363d',
                'linecolor': '#30363d',
                'tickcolor': '#30363d',
                'color': '#8b949e'
            },
            'yaxis': {
                'gridcolor': '#30363d',
                'linecolor': '#30363d',
                'tickcolor': '#30363d',
                'color': '#8b949e'
            },
            'legend': {
                'font': {'color': '#f0f6fc'}
            },
            'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
        }
    }

def create_investment_chart(df):
    """Create investment grade distribution chart with dark theme"""
    if 'investment_grade' not in df.columns:
        st.warning("⚠️ Investment grade data not available")
        return
    
    grade_counts = df['investment_grade'].value_counts()
    
    # Dark theme colors
    colors = {
        'Strong Buy': '#3fb950',
        'Buy': '#58a6ff',
        'Hold': '#d29922',
        'Avoid': '#f85149',
        'Weak Hold': '#bc8cff'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=grade_counts.index,
        values=grade_counts.values,
        hole=0.6,
        marker_colors=[colors.get(grade, '#8b949e') for grade in grade_counts.index],
        textinfo='label+percent',
        textposition='outside',
        textfont={'size': 14, 'color': '#f0f6fc'},
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])
    
    # Apply dark theme configuration
    theme_config = get_dark_theme_config()
    fig.update_layout(
        paper_bgcolor=theme_config['layout']['paper_bgcolor'],
        plot_bgcolor=theme_config['layout']['plot_bgcolor'],
        font=theme_config['layout']['font'],
        title={
            'text': '🎯 Investment Grade Distribution',
            'font': {'color': '#f0f6fc', 'size': 18, 'family': 'Inter'},
            'x': 0.5
        },
        height=500,
        showlegend=True,
        legend={
            'orientation': 'h',
            'y': -0.1,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'color': '#f0f6fc', 'size': 12}
        },
        margin={'l': 40, 'r': 40, 't': 60, 'b': 40}
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_stock_cards(df, limit=5):
    """Create enhanced stock cards"""
    st.html('<div class="sub-header">🌟 Top Investment Opportunities</div>')
    
    # Sort by value_score if available
    if 'value_score' in df.columns:
        top_stocks = df.nlargest(limit, 'value_score')
    else:
        top_stocks = df.head(limit)
    
    for idx, stock in top_stocks.iterrows():
        symbol = stock.get('symbol', 'N/A')
        company_name = stock.get('company_name', 'N/A')
        current_price = stock.get('current_price', 0)
        target_price = stock.get('target_price', current_price)
        value_score = stock.get('value_score', 0)
        investment_grade = stock.get('investment_grade', 'N/A')
        
        # Calculate upside safely
        if current_price > 0 and pd.notna(target_price) and str(target_price).lower() != 'n/a':
            try:
                target_price_num = float(target_price)
                upside = ((target_price_num - current_price) / current_price) * 100
            except (ValueError, TypeError):
                upside = 0
        else:
            upside = 0
        
        # Grade styling
        if 'Strong Buy' in str(investment_grade):
            grade_class = "grade-strong-buy"
        elif 'Buy' in str(investment_grade):
            grade_class = "grade-buy"
        else:
            grade_class = "grade-hold"
        
        # Format values safely with proper styling
        safe_score = f"{value_score:.1f}" if pd.notna(value_score) else "N/A"
        safe_current = format_currency(current_price)
        safe_target = format_currency(target_price)
        safe_upside = f"{upside:+.1f}%" if upside != 0 else "N/A"
        
        # Determine upside styling class
        if safe_upside == "N/A":
            upside_class = "metrics-current"  # Neutral color for N/A
        elif upside > 0:
            upside_class = "metrics-upside"   # Green for positive
        else:
            upside_class = "metrics-negative" # Red for negative
        
        html_content = f"""
        <div class="metric-card" style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">{symbol}</h3>
                    <p style="margin: 0.25rem 0 0 0; color: #7f8c8d;">{company_name}</p>
                </div>
                <div class="{grade_class}">
                    {investment_grade}
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metrics-grid-item">
                    <div class="metrics-label">Score</div>
                    <div class="metrics-value metrics-score">{safe_score}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Current</div>
                    <div class="metrics-value metrics-current">{safe_current}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Target</div>
                    <div class="metrics-value metrics-target">{safe_target}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Upside</div>
                    <div class="metrics-value {upside_class}">{safe_upside}</div>
                </div>
            </div>
        </div>
        """
        st.html(html_content)

def create_performance_charts(df):
    """Create performance analysis charts"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price vs Target', 'Value Score Distribution', 'Market Cap Analysis', 'PE Ratio Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price vs Target scatter
    if 'current_price' in df.columns and 'target_price' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['current_price'],
                y=df['target_price'],
                mode='markers',
                text=df.get('symbol', ''),
                name='Price vs Target'
            ),
            row=1, col=1
        )
    
    # Value Score distribution
    if 'value_score' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['value_score'], name='Value Score'),
            row=1, col=2
        )
    
    # Market Cap analysis
    if 'market_cap' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['market_cap'], name='Market Cap'),
            row=2, col=1
        )
    
    # PE Ratio analysis  
    if 'pe_ratio' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['pe_ratio'], name='PE Ratio'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        title_text="📊 Performance Analytics Dashboard",
        title_x=0.5,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_detailed_stock_analysis(df):
    """Create detailed analysis cards for recommended stocks"""
    for idx, stock in df.iterrows():
        symbol = stock.get('symbol', 'N/A')
        company_name = stock.get('company_name', 'N/A')
        current_price = stock.get('current_price', 0)
        target_price = stock.get('target_price', current_price)
        investment_grade = stock.get('investment_grade', 'N/A')
        value_score = stock.get('value_score', 0)
        
        # Get live price data
        live_price, price_change, change_pct = get_current_price(symbol)
        
        # Calculate potential upside
        upside = ((target_price - (live_price or current_price)) / (live_price or current_price)) * 100 if (live_price or current_price) > 0 else 0
        
        # Create analysis card
        with st.expander(f"📊 {symbol} - {company_name}", expanded=False):
            # Main metrics grid
            safe_score = f"{value_score:.1f}" if pd.notna(value_score) else "N/A"
            safe_current = format_currency(live_price or current_price)
            safe_target = format_currency(target_price)  
            safe_upside = f"{upside:+.1f}%" if pd.notna(upside) and upside != 0 else "N/A"
            
            # Determine upside styling class
            if safe_upside == "N/A":
                upside_class = "metrics-current"  # Neutral color for N/A
            elif upside > 0:
                upside_class = "metrics-upside"   # Green for positive
            else:
                upside_class = "metrics-negative" # Red for negative
            
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metrics-grid-item">
                    <div class="metrics-label">Score</div>
                    <div class="metrics-value metrics-score">{safe_score}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Current</div>
                    <div class="metrics-value metrics-current">{safe_current}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Target</div>
                    <div class="metrics-value metrics-target">{safe_target}</div>
                </div>
                <div class="metrics-grid-item">
                    <div class="metrics-label">Upside</div>
                    <div class="metrics-value {upside_class}">{safe_upside}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional details in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 💰 Live Data")
                if live_price and price_change is not None:
                    price_color = "price-positive" if price_change >= 0 else "price-negative"
                    st.markdown(f"""
                    <div class="price-card">
                        <strong>Live Price:</strong> <span class="{price_color}">{format_currency(live_price)}</span><br>
                        <strong>Change:</strong> <span class="{price_color}">{format_currency(price_change)} ({format_percentage(change_pct)})</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Live data not available")
            
            with col2:
                st.markdown("#### 📈 Investment Metrics")
                rec_style = get_recommendation_style(investment_grade)
                if rec_style:
                    st.markdown(f'**Grade:** <span class="{rec_style}">{investment_grade}</span>', 
                              unsafe_allow_html=True)
                else:
                    st.write(f"**Grade:** {investment_grade}")
                
                st.write(f"**Value Score:** {value_score:.1f}/10")
                st.write(f"**PE Ratio:** {stock.get('pe_ratio', 'N/A')}")
                st.write(f"**ROE:** {stock.get('roe', 'N/A')}%")
            
            with col3:
                st.markdown("#### 🏢 Company Info")
                st.write(f"**Market Cap:** ₹{stock.get('market_cap', 0):.0f} Cr")
                st.write(f"**Sector:** {stock.get('sector', 'N/A')}")
                st.write(f"**Industry:** {stock.get('industry', 'N/A')}")
                
                # AI Analysis summary if available
                if 'ai_analysis' in stock and pd.notna(stock['ai_analysis']):
                    st.markdown("**AI Analysis:**")
                    st.info(str(stock['ai_analysis'])[:200] + "..." if len(str(stock['ai_analysis'])) > 200 else str(stock['ai_analysis']))

def create_price_prediction_chart(stock_data):
    """Create comprehensive price prediction analysis for individual stock"""
    st.markdown("#### 📈 Advanced Price Analysis & Predictions")
    
    symbol = stock_data.get('symbol', 'N/A')
    current_price = stock_data.get('current_price', 0)
    target_price = stock_data.get('target_price', current_price)
    
    # Check if symbol already has .NS suffix
    if not symbol.endswith('.NS'):
        ticker_symbol = f"{symbol}.NS"
    else:
        ticker_symbol = symbol
    
    # If price prediction service is not available, use basic chart
    if not HAS_PRICE_PREDICTION:
        st.error("� AI Prediction Service Not Available")
        st.markdown("""
        **AI predictions are currently disabled:**
        - Required prediction service components are not installed
        - System configuration issue detected
        - Service dependencies are missing
        
        **Available data:**
        - ✅ Historical price charts and trends
        - ✅ Basic market data and company information
        - ✅ Current price and volume data
        
        **To enable predictions:**
        - Contact your system administrator
        - Check installation requirements
        - Verify service configuration
        """)
        
        create_basic_price_chart(stock_data, ticker_symbol)
        return
    
    # Initialize prediction service
    try:
        # Use new modular prediction system
        orchestrator = PricePredictionOrchestrator(ticker_symbol)
        current_price = orchestrator._get_current_price()
        
        # Validate that we have valid price data before proceeding
        if current_price is None or current_price <= 0:
            st.error("🚫 Price Data Unavailable")
            st.markdown(f"""
            **Cannot generate predictions for {symbol}:**
            - Stock price data is not available (possibly delisted)
            - Current price: {current_price}
            - Data source connectivity issues
            
            **Recommendations:**
            - ✅ Try a different stock symbol
            - 🔄 Check if the stock is still actively traded
            - 📊 Use the main analysis tab for available data
            """)
            return
        
        # Get comprehensive predictions
        with st.spinner(f"Analyzing {symbol} and generating AI predictions..."):
            # Get main prediction (30-day default)
            predictions = orchestrator.predict_comprehensive(target_days=30)
            
            # Get multi-period predictions (7d, 30d, 90d, 180d, 365d)
            multi_period = {
                '7d': orchestrator.predict_comprehensive(target_days=7),
                '30d': predictions,  # Use the already computed 30-day prediction
                '90d': orchestrator.predict_comprehensive(target_days=90),
                '180d': orchestrator.predict_comprehensive(target_days=180),  # 6 months
                '365d': orchestrator.predict_comprehensive(target_days=365)
            }
        
        if "error" not in predictions:
            # Create tabs for different prediction views
            pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
                "📊 Price Chart", "🔮 Predictions", "📈 Multi-Period", "⚠️ Risk Analysis"
            ])
            
            with pred_tab1:
                # Historical price chart with predictions
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="6mo")
                
                if not hist.empty:
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    # Current price line
                    fig.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color="#e74c3c",
                        annotation_text=f"Current: ₹{current_price:.2f}"
                    )
                    
                    # Target price line
                    if target_price > 0:
                        fig.add_hline(
                            y=target_price,
                            line_dash="dot",
                            line_color="#27ae60",
                            annotation_text=f"Target: ₹{target_price:.2f}"
                        )
                    
                    # Ensemble prediction line
                    if 'ensemble' in predictions and 'predicted_price' in predictions['ensemble']:
                        ensemble_price = predictions['ensemble']['predicted_price']
                        fig.add_hline(
                            y=ensemble_price,
                            line_dash="dashdot",
                            line_color="#9b59b6",
                            annotation_text=f"AI Prediction: ₹{ensemble_price:.2f}"
                        )
                    
                    fig.update_layout(
                        title=f"{symbol} - Price Analysis & Predictions",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("52W High", f"₹{hist['Close'].max():.2f}")
                    with col2:
                        st.metric("52W Low", f"₹{hist['Close'].min():.2f}")
                    with col3:
                        volatility = hist['Close'].pct_change().std() * 100
                        st.metric("Volatility", f"{volatility:.1f}%")
                    with col4:
                        avg_volume = hist['Volume'].mean()
                        st.metric("Avg Volume", f"{avg_volume/1000000:.1f}M")
            
            with pred_tab2:
                st.markdown("##### 🎯 Individual Prediction Methods")
                
                # Time horizon information - more concise
                st.info("⏰ **Time Horizons:** Short-term predictions are for 30 days. Fundamental shows intrinsic fair value (6-12 months horizon).")
                
                # Create prediction comparison using new modular structure
                individual_preds = predictions.get('individual_predictions', {})
                
                if individual_preds:
                    # Display predictions in a balanced 3-column layout
                    st.markdown("**🎯 AI Prediction Methods Overview**")
                    
                    # First row: Technical, ML, Volume
                    pred_cols_row1 = st.columns(3)
                    
                    with pred_cols_row1[0]:
                        st.markdown("**📈 Technical Analysis**")
                        if 'technical' in individual_preds:
                            tech = individual_preds['technical']
                            if 'error' not in tech and 'predicted_price' in tech:
                                st.metric(
                                    "30-Day Price",
                                    f"₹{tech['predicted_price']:.2f}",
                                    delta=f"{((tech['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption(f"📊 Confidence: {tech.get('confidence', 'N/A')}")
                            else:
                                st.error("❌ Model Error")
                    
                    with pred_cols_row1[1]:
                        st.markdown("**🤖 Machine Learning**")
                        if 'machine_learning' in individual_preds:
                            ml = individual_preds['machine_learning']
                            if 'error' not in ml and 'predicted_price' in ml:
                                st.metric(
                                    "30-Day Price",
                                    f"₹{ml['predicted_price']:.2f}",
                                    delta=f"{((ml['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption(f"🔧 Model: {ml.get('best_model', 'N/A')}")
                            else:
                                st.error("❌ Model Error")
                    
                    with pred_cols_row1[2]:
                        st.markdown("**📊 Volume Analysis**")
                        if 'volume_analysis' in individual_preds:
                            vol = individual_preds['volume_analysis']
                            if 'error' not in vol and 'predicted_price' in vol:
                                st.metric(
                                    "30-Day Price",
                                    f"₹{vol['predicted_price']:.2f}",
                                    delta=f"{((vol['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption("📈 Volume-based prediction")
                            else:
                                st.error("❌ Model Error")
                    
                    st.markdown("---")  # Visual separator
                    
                    # Second row: Fundamental, Time Series, Pattern
                    pred_cols_row2 = st.columns(3)
                    
                    with pred_cols_row2[0]:
                        st.markdown("**💰 Fundamental Analysis**")
                        if 'fundamental' in individual_preds:
                            fund = individual_preds['fundamental']
                            if 'error' not in fund and 'predicted_price' in fund:
                                st.metric(
                                    "Fair Value",
                                    f"₹{fund['predicted_price']:.2f}",
                                    delta=f"{((fund['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption("⏰ Long-term (6-12 months)")
                            else:
                                st.error("❌ Model Error")
                    
                    with pred_cols_row2[1]:
                        st.markdown("**📈 Time Series**")
                        if 'time_series' in individual_preds:
                            ts = individual_preds['time_series']
                            if 'error' not in ts and 'predicted_price' in ts:
                                st.metric(
                                    "30-Day Price",
                                    f"₹{ts['predicted_price']:.2f}",
                                    delta=f"{((ts['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption(f"📊 ARIMA & Time Series")
                            else:
                                st.error("❌ Model Error")
                    
                    with pred_cols_row2[2]:
                        st.markdown("**🔍 Pattern Recognition**")
                        if 'pattern_recognition' in individual_preds:
                            pattern = individual_preds['pattern_recognition']
                            if 'error' not in pattern and 'predicted_price' in pattern:
                                st.metric(
                                    "30-Day Price",
                                    f"₹{pattern['predicted_price']:.2f}",
                                    delta=f"{((pattern['predicted_price']/current_price - 1) * 100):+.1f}%"
                                )
                                st.caption("🔍 Chart pattern analysis")
                            else:
                                st.error("❌ Model Error")
                
                # Main Ensemble Prediction (30-day)
                if 'predicted_price' in predictions:
                    st.markdown("---")
                    st.markdown("##### 🎯 **Final AI Prediction (30-Day Ensemble Forecast)**")
                    
                    ensemble_price = predictions['predicted_price']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "AI Predicted Price",
                            f"₹{ensemble_price:.2f}",
                            delta=f"{((ensemble_price/current_price - 1) * 100):+.1f}%"
                        )
                    with col2:
                        confidence = predictions.get('confidence', 0)
                        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                        st.metric(
                            "Confidence Score",
                            f"{confidence_color} {confidence:.1%}"
                        )
                    with col3:
                        models_used = predictions.get('models_used', 0)
                        st.metric(
                            "Models Used",
                            f"⚙️ {models_used}/6"
                        )
            
            with pred_tab3:
                st.markdown("##### 📈 Multi-Period Price Forecasts")
                
                # Check if we have multi-period prediction data
                # The multi_period structure is {'7d': predictions, '30d': predictions, ...}
                periods_data = multi_period
                
                if periods_data and len(periods_data) > 0:
                    # Create comprehensive dataframe for multi-period predictions
                    period_data = []
                    for period_name, period_data_dict in periods_data.items():
                        # Check if this period has valid predictions (no error)
                        if isinstance(period_data_dict, dict) and 'error' not in period_data_dict and 'predicted_price' in period_data_dict:
                            predicted_price = period_data_dict['predicted_price']
                            change_pct = ((predicted_price/current_price - 1) * 100)
                            
                            period_data.append({
                                'Period': period_name.replace('d', ' days').replace('30 days', '1 Month').replace('90 days', '3 Months').replace('180 days', '6 Months').replace('365 days', '12 Months').replace('7 days', '1 Week'),  # Format period names
                                'Predicted Price': predicted_price,
                                'Target Price': target_price,  # Add target price
                                'Price Change %': change_pct,   # Price change %
                                'Confidence': period_data_dict.get('confidence', 0)
                            })
                    
                    if period_data:
                        df_periods = pd.DataFrame(period_data)
                        
                        # Enhanced Key Metrics Section
                        st.markdown("**📊 Key Growth & Target Metrics**")
                        
                        # Find 6M and 12M data specifically
                        six_month_data = None
                        twelve_month_data = None
                        
                        for _, row in df_periods.iterrows():
                            if '6 Months' in str(row['Period']):
                                six_month_data = row
                            elif '12 Months' in str(row['Period']):
                                twelve_month_data = row
                        
                        # Create enhanced metrics display
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            if six_month_data is not None:
                                six_month_growth = six_month_data['Price Change %']
                                growth_color = "🟢" if six_month_growth > 0 else "🔴"
                                st.metric(
                                    "6M Growth",
                                    f"{growth_color} {six_month_growth:+.1f}%",
                                    delta=f"₹{six_month_data['Predicted Price'] - current_price:.2f}"
                                )
                            else:
                                st.metric("6M Growth", "N/A")
                        
                        with metric_cols[1]:
                            if twelve_month_data is not None:
                                twelve_month_growth = twelve_month_data['Price Change %']
                                growth_color = "🟢" if twelve_month_growth > 0 else "🔴"
                                st.metric(
                                    "12M Growth",
                                    f"{growth_color} {twelve_month_growth:+.1f}%",
                                    delta=f"₹{twelve_month_data['Predicted Price'] - current_price:.2f}"
                                )
                            else:
                                st.metric("12M Growth", "N/A")
                        
                        with metric_cols[2]:
                            if six_month_data is not None:
                                st.metric(
                                    "6M Price Target",
                                    f"₹{six_month_data['Predicted Price']:.2f}",
                                    delta=f"vs Current: {six_month_data['Price Change %']:+.1f}%"
                                )
                            else:
                                st.metric("6M Price Target", "N/A")
                        
                        with metric_cols[3]:
                            # Overall price change % (current vs target)
                            if target_price > 0:
                                target_change = ((target_price/current_price - 1) * 100)
                                change_color = "🟢" if target_change > 0 else "🔴"
                                st.metric(
                                    "Price Change % (Target)",
                                    f"{change_color} {target_change:+.1f}%",
                                    delta=f"₹{target_price - current_price:.2f}"
                                )
                            else:
                                st.metric("Price Change % (Target)", "N/A")
                        
                        with metric_cols[4]:
                            # AI Fair Value
                            if target_price > 0:
                                st.metric(
                                    "AI Fair Value (12M)",
                                    f"₹{target_price:.2f}",
                                    delta=f"vs Current: {((target_price/current_price - 1) * 100):+.1f}%",
                                    help="AI-generated fair value estimate (12-month horizon)"
                                )
                            else:
                                st.metric("AI Fair Value (12M)", "N/A")
                        
                        st.markdown("---")
                        
                        # Detailed period-by-period forecasts
                        st.markdown("**📅 Period-by-Period Forecasts**")
                        
                        # Display as metrics with enhanced information
                        cols = st.columns(min(len(df_periods), 6))  # Limit to 6 columns for better display
                        for i, (_, row) in enumerate(df_periods.iterrows()):
                            if i < 6:  # Only show first 6 periods
                                with cols[i]:
                                    confidence_emoji = "🟢" if row['Confidence'] > 0.7 else "🟡" if row['Confidence'] > 0.5 else "🔴"
                                    change_emoji = "📈" if row['Price Change %'] > 0 else "📉"
                                    
                                    st.metric(
                                        f"{change_emoji} {row['Period']}",
                                        f"₹{row['Predicted Price']:.2f}",
                                        delta=f"{row['Price Change %']:+.1f}%"
                                    )
                                    st.caption(f"{confidence_emoji} {row['Confidence']:.1%} confidence")
                        
                        # Enhanced comparison table
                        st.markdown("**📋 Detailed Forecast Table**")
                        
                        # Create enhanced display dataframe
                        display_df = df_periods.copy()
                        display_df['Predicted Price'] = display_df['Predicted Price'].apply(lambda x: f"₹{x:.2f}")
                        display_df['Target Price'] = display_df['Target Price'].apply(lambda x: f"₹{x:.2f}" if x > 0 else "N/A")
                        display_df['Price Change %'] = display_df['Price Change %'].apply(lambda x: f"{x:+.1f}%")
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                        
                        # Add risk level based on change %
                        display_df['Risk Level'] = df_periods['Price Change %'].apply(
                            lambda x: "🟢 Low" if abs(x) < 15 
                            else "🟡 Medium" if abs(x) < 30 
                            else "🔴 High"
                        )
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Enhanced chart for multi-period predictions
                        fig_multi = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Price Predictions', 'Growth % by Period'),
                            vertical_spacing=0.1
                        )
                        
                        # Price prediction chart
                        fig_multi.add_trace(
                            go.Scatter(
                                x=df_periods['Period'],
                                y=df_periods['Predicted Price'],
                                mode='lines+markers',
                                name='AI Predictions',
                                line=dict(color='#9b59b6', width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Price: ₹%{y:.2f}<extra></extra>'
                            ), row=1, col=1
                        )
                        
                        # Add current price baseline
                        fig_multi.add_hline(
                            y=current_price,
                            line_dash="dash",
                            line_color="#e74c3c",
                            annotation_text=f"Current: ₹{current_price:.2f}",
                            row=1, col=1
                        )
                        
                        # Add target price line if available
                        if target_price > 0:
                            fig_multi.add_hline(
                                y=target_price,
                                line_dash="dot",
                                line_color="#27ae60",
                                annotation_text=f"Analyst Target: ₹{target_price:.2f}",
                                row=1, col=1
                            )
                        
                        # Growth percentage chart
                        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in df_periods['Price Change %']]
                        fig_multi.add_trace(
                            go.Bar(
                                x=df_periods['Period'],
                                y=df_periods['Price Change %'],
                                name='Growth %',
                                marker_color=colors,
                                hovertemplate='<b>%{x}</b><br>Growth: %{y:+.1f}%<extra></extra>'
                            ), row=2, col=1
                        )
                        
                        fig_multi.update_layout(
                            title=f"{symbol} - Comprehensive Multi-Period Analysis",
                            height=600,
                            showlegend=True
                        )
                        
                        fig_multi.update_xaxes(title_text="Time Period", row=1, col=1)
                        fig_multi.update_yaxes(title_text="Price (₹)", row=1, col=1)
                        fig_multi.update_xaxes(title_text="Time Period", row=2, col=1)
                        fig_multi.update_yaxes(title_text="Growth %", row=2, col=1)
                        
                        st.plotly_chart(fig_multi, use_container_width=True)
                        
                        # Key insights section
                        st.markdown("**🔍 Key Insights**")
                        
                        insights_col1, insights_col2 = st.columns(2)
                        
                        with insights_col1:
                            st.markdown("**Growth Analysis:**")
                            max_growth_period = df_periods.loc[df_periods['Price Change %'].idxmax()]
                            min_growth_period = df_periods.loc[df_periods['Price Change %'].idxmin()]
                            
                            st.write(f"• **Best Period**: {max_growth_period['Period']} (+{max_growth_period['Price Change %']:.1f}%)")
                            st.write(f"• **Weakest Period**: {min_growth_period['Period']} ({min_growth_period['Price Change %']:+.1f}%)")
                            
                            avg_growth = df_periods['Price Change %'].mean()
                            st.write(f"• **Average Growth**: {avg_growth:+.1f}%")
                        
                        with insights_col2:
                            st.markdown("**Risk & Confidence:**")
                            high_confidence_periods = len(df_periods[df_periods['Confidence'] > 0.7])
                            st.write(f"• **High Confidence Periods**: {high_confidence_periods}/{len(df_periods)}")
                            
                            avg_confidence = df_periods['Confidence'].mean()
                            confidence_color = "🟢" if avg_confidence > 0.7 else "🟡" if avg_confidence > 0.5 else "🔴"
                            st.write(f"• **Average Confidence**: {confidence_color} {avg_confidence:.1%}")
                            
                            if target_price > 0:
                                target_achievable = any(abs(row['Predicted Price'] - target_price) / target_price < 0.1 for _, row in df_periods.iterrows())
                                st.write(f"• **Target Achievable**: {'🟢 Likely' if target_achievable else '🟡 Uncertain'}")
                    
                    else:
                        # Fallback when period_data is empty
                        create_fallback_multi_period_forecast(symbol, current_price, target_price)
                
                else:
                    # Fallback when no multi-period data is available
                    create_fallback_multi_period_forecast(symbol, current_price, target_price)
            
            with pred_tab4:
                st.markdown("##### ⚠️ Risk Assessment & Model Reliability")
                
                risk_info = predictions.get('risk_assessment', {})
                
                if risk_info:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prediction Risk Metrics**")
                        
                        volatility_risk = risk_info.get('volatility_risk', 0)
                        risk_color = "🟢" if volatility_risk < 0.3 else "🟡" if volatility_risk < 0.6 else "🔴"
                        st.metric("Volatility Risk", f"{risk_color} {volatility_risk:.1%}")
                        
                        model_agreement = risk_info.get('model_agreement', 0)
                        agreement_color = "🟢" if model_agreement > 0.7 else "🟡" if model_agreement > 0.5 else "🔴"
                        st.metric("Model Agreement", f"{agreement_color} {model_agreement:.1%}")
                        
                        data_quality = risk_info.get('data_quality_score', 0)
                        quality_color = "🟢" if data_quality > 0.8 else "🟡" if data_quality > 0.6 else "🔴"
                        st.metric("Data Quality", f"{quality_color} {data_quality:.1%}")
                    
                    with col2:
                        st.markdown("**Investment Recommendations**")
                        
                        overall_risk = risk_info.get('overall_risk_level', 'Unknown')
                        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(overall_risk, "⚪")
                        st.info(f"**Overall Risk Level:** {risk_emoji} {overall_risk}")
                        
                        if 'recommendation' in risk_info:
                            st.success(f"**Recommendation:** {risk_info['recommendation']}")
                        
                        if 'key_risks' in risk_info:
                            st.markdown("**Key Risk Factors:**")
                            for risk in risk_info['key_risks']:
                                st.write(f"• {risk}")
                
                # Model performance metrics
                if 'ensemble' in predictions and 'model_metrics' in predictions['ensemble']:
                    st.markdown("---")
                    st.markdown("**Model Performance Metrics**")
                    
                    metrics = predictions['ensemble']['model_metrics']
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        if 'mae' in metrics:
                            st.metric("Mean Absolute Error", f"₹{metrics['mae']:.2f}")
                    with metric_cols[1]:
                        if 'rmse' in metrics:
                            st.metric("Root Mean Squared Error", f"₹{metrics['rmse']:.2f}")
                    with metric_cols[2]:
                        if 'accuracy_score' in metrics:
                            st.metric("Accuracy Score", f"{metrics['accuracy_score']:.1%}")
        
        else:
            st.error("🚫 AI Prediction Service Currently Unavailable")
            st.markdown(f"""
            **Unable to generate predictions for {symbol}:**
            - AI prediction models are temporarily offline
            - Data connectivity issues detected
            - Service is under maintenance
            
            **Available alternatives:**
            - ✅ Check historical price charts below
            - ✅ View basic company metrics in other tabs
            - 🔄 Try again in a few minutes
            """)
            
            # Show basic price chart as alternative
            create_basic_price_chart(stock_data, ticker_symbol)
    
    except Exception as e:
        st.error("🚫 AI Prediction System Error")
        st.markdown(f"""
        **Technical error occurred:**
        - System: `{str(e)}`
        - Status: Prediction service unavailable
        
        **What you can do:**
        - ✅ Basic price charts are still available below
        - 🔄 Refresh the page and try again
        - 📞 Contact support if issue persists
        """)
        
        # Provide basic price analysis as fallback
        create_basic_price_chart(stock_data, ticker_symbol)

def create_fallback_multi_period_forecast(symbol, current_price, target_price):
    """Create an honest fallback when prediction data is not available"""
    # Status indicator with clear messaging
    st.error("🚫 Multi-Period Prediction Data Not Available")
    
    # Clear explanation of why this happened
    st.markdown("""
    **Why AI predictions are unavailable:**
    - 🔧 AI prediction service is currently under maintenance
    - 📡 Network connectivity issues affecting data retrieval
    - 📊 Insufficient recent market data for reliable forecasting
    - ⚡ Service temporarily overloaded - please try again shortly
    """)
    
    # Show what current data IS available
    st.info(f"💡 **Available Data for {symbol}**: Current market price and basic information are accessible below.")
    
    # Display basic current information (no projections/predictions)
    try:
        current_price_float = float(current_price) if current_price > 0 else 0
        target_price_float = float(target_price) if target_price > 0 else 0
        
        # Show only factual current data - no predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if current_price_float > 0:
                st.metric(
                    "Current Market Price", 
                    f"₹{current_price_float:.2f}",
                    help="Last available trading price"
                )
            else:
                st.metric("Current Market Price", "Data unavailable")
                
        with col2:
            if target_price_float > 0:
                change_pct = ((target_price_float / current_price_float) - 1) * 100 if current_price_float > 0 else 0
                st.metric(
                    "AI Fair Value (12M)", 
                    f"₹{target_price_float:.2f}",
                    delta=f"{change_pct:+.1f}%" if current_price_float > 0 else None,
                    help="AI-generated fair value estimate (12-month horizon)"
                )
            else:
                st.metric("AI Fair Value (12M)", "Not available")
                
        with col3:
            st.metric(
                "Prediction Status", 
                "❌ Offline",
                help="AI prediction service is currently unavailable"
            )
        
    except Exception as e:
        st.error(f"Unable to display current data: {str(e)}")
    
    st.markdown("---")
    
    # Alternative suggestions for users
    st.markdown("""
    **🔄 What you can do while predictions are unavailable:**
    
    ✅ **Available Now:**
    - Check the **Price Chart** tab for historical price trends
    - Review the **Predictions** tab for any available short-term forecasts
    - Use the **Risk Analysis** tab for volatility insights
    - Access current market price and basic company information
    
    🔄 **Retry Options:**
    - Refresh the page in 2-3 minutes
    - Select a different stock and return to this one
    - Check your internet connection
    
    📞 **Need Help?**
    - Contact support if predictions remain unavailable for extended periods
    - Check system status updates on our main page
    """)
    
    # Honest disclaimer - no fake predictions
    st.warning("""
    **⚠️ Important**: We do not provide mock or simulated predictions when our AI service is unavailable. 
    All forecasts will return once the service is restored to ensure you receive only accurate, 
    data-driven predictions.
    """)

def create_basic_price_chart(stock_data, ticker_symbol):
    """Fallback function for basic price chart when predictions fail"""
    symbol = stock_data.get('symbol', 'N/A')
    current_price = stock_data.get('current_price', 0)
    target_price = stock_data.get('target_price', current_price)
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="6mo")
        
        if not hist.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Current: ₹{current_price:.2f}"
            )
            
            if target_price > 0:
                fig.add_hline(
                    y=target_price,
                    line_dash="dot",
                    line_color="#27ae60",
                    annotation_text=f"Target: ₹{target_price:.2f}"
                )
            
            fig.update_layout(
                title=f"{symbol} - Basic Price Analysis",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical price data available")
    
    except Exception as e:
        st.error(f"Could not load basic price chart: {str(e)}")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Current Price', 'Target Price'],
            y=[current_price, target_price],
            marker_color=['#3498db', '#27ae60']
        ))
        
        fig.update_layout(
            title=f"{symbol} - Price Comparison",
            yaxis_title="Price (₹)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_ai_insights(df):
    """Show AI insights section"""
    st.html('<div class="sub-header">🤖 AI Analysis Insights</div>')
    
    # Business Quality distribution
    if 'business_quality' in df.columns:
        # Convert to numeric if needed
        business_quality = pd.to_numeric(df['business_quality'], errors='coerce').fillna(50)
        
        fig = px.histogram(business_quality, nbins=20, title='Business Quality Score Distribution')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Reasoning for top stocks
    if 'ai_reasoning' in df.columns and 'value_score' in df.columns:
        st.markdown("#### 💡 AI Analysis for Top Stocks")
        
        top_3_stocks = df.nlargest(3, 'value_score')
        for _, stock in top_3_stocks.iterrows():
            with st.expander(f"🔍 {stock['symbol']} - AI Analysis"):
                reasoning = stock.get('ai_reasoning', 'No AI analysis available')
                if pd.notna(reasoning) and str(reasoning).strip():
                    st.markdown(str(reasoning))
                else:
                    st.markdown("*AI analysis not available for this stock.*")

def show_modern_status_bar():
    """Display modern status bar with system information"""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Create status bar with modern design
    status_html = f"""
    <div style="
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0 2rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    ">
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <span class="status-online">
                <span style="
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: var(--success);
                    border-radius: 50%;
                    margin-right: 0.5rem;
                    animation: pulse 2s infinite;
                "></span>
                AI System Online
            </span>
            <span class="status-online">
                <span style="
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: var(--success);
                    border-radius: 50%;
                    margin-right: 0.5rem;
                    animation: pulse 2s infinite;
                "></span>
                Data Feed Live
            </span>
        </div>
        <div style="
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
        ">
            Last Updated: {current_time}
        </div>
    </div>
    """
    
    st.html(status_html)

def show_dashboard_header():
    """Display modern dashboard header with enhanced styling"""
    header_html = f"""
    <div style="text-align: center; margin-bottom: 3rem;">
        <div class="main-header">
            🚀 AI Stock Screener Pro
        </div>
        <div style="
            font-size: 1.125rem;
            color: var(--text-secondary);
            font-weight: 500;
            margin-top: 1rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        ">
            Powered by Advanced AI • Real-time Analysis • Professional Grade Insights
        </div>
    </div>
    """
    
    st.html(header_html)

def main():
    """Main dashboard function with modern dark theme"""
    # Modern header
    show_dashboard_header()
    
    # Status bar
    show_modern_status_bar()
    
    # Load data
    df, message = load_data()
    
    if df is None:
        st.error(f"❌ {message}")
        st.info("📋 Please run the screener first: `python scripts/run_ai_screener.py`")
        return
    
    st.success(f"✅ {message}")
    
    # Modern sidebar with enhanced styling
    with st.sidebar:
        st.html("""
        <div style="
            text-align: center;
            padding: 2rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid var(--border-primary);
        ">
            <div style="
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
            ">🎛️ Analysis Filters</div>
            <div style="
                font-size: 0.875rem;
                color: var(--text-secondary);
            ">Customize your investment analysis</div>
        </div>
        """)
    
    # Investment grade filter
    if 'investment_grade' in df.columns:
        unique_grades = df['investment_grade'].dropna().unique()
        selected_grades = st.sidebar.multiselect(
            "Investment Grades",
            options=unique_grades,
            default=unique_grades
        )
        if selected_grades:
            df = df[df['investment_grade'].isin(selected_grades)]
    
    # Value score filter
    if 'value_score' in df.columns:
        min_score = st.sidebar.slider(
            "Minimum Value Score",
            min_value=float(df['value_score'].min()),
            max_value=float(df['value_score'].max()),
            value=float(df['value_score'].min())
        )
        df = df[df['value_score'] >= min_score]
    
    # Main tabs - Adding Investment Recommendations tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Performance", "🤖 AI Insights", "🏆 Investment Recommendations", "🔍 Stock Analysis", "📋 Data Table"
    ])
    
    with tab1:
        st.markdown("### 📊 Market Overview")
        
        # Enhanced overview with current prices
        col1, col2 = st.columns([2, 1])
        
        with col1:
            create_modern_summary_metrics(df)
        
        with col2:
            st.markdown("#### 🕒 Live Market Data")
            st.markdown("*Last updated: " + datetime.now().strftime("%H:%M:%S") + "*")
            
        create_investment_chart(df)
        create_stock_cards(df)
    
    with tab2:
        st.markdown("### 📈 Performance Analytics")
        create_performance_charts(df)
    
    with tab3:
        st.markdown("### 🤖 AI Insights")
        show_ai_insights(df)
    
    with tab4:
        st.markdown("### 🏆 Investment Recommendations")
        
        # Filter for recommended stocks
        recommended_df = df.copy()
        if 'investment_grade' in df.columns:
            recommended_stocks = recommended_df[
                recommended_df['investment_grade'].str.contains('Buy|Strong Buy', case=False, na=False)
            ]
        else:
            recommended_stocks = recommended_df.head(10)  # Top 10 if no grades
            
        if not recommended_stocks.empty:
            st.markdown("#### 🌟 Top Investment Picks")
            
            # Create tabs for different recommendation levels
            rec_tabs = st.tabs(["🚀 Strong Buy", "📈 Buy", "📊 All Recommendations"])
            
            with rec_tabs[0]:
                strong_buy = recommended_stocks[
                    recommended_stocks['investment_grade'].str.contains('Strong Buy', case=False, na=False)
                ] if 'investment_grade' in recommended_stocks.columns else pd.DataFrame()
                
                if not strong_buy.empty:
                    st.success(f"Found {len(strong_buy)} Strong Buy recommendations")
                    create_detailed_stock_analysis(strong_buy)
                else:
                    st.info("No Strong Buy recommendations at this time.")
            
            with rec_tabs[1]:
                buy_stocks = recommended_stocks[
                    recommended_stocks['investment_grade'].str.contains('Buy', case=False, na=False) &
                    ~recommended_stocks['investment_grade'].str.contains('Strong Buy', case=False, na=False)
                ] if 'investment_grade' in recommended_stocks.columns else pd.DataFrame()
                
                if not buy_stocks.empty:
                    st.success(f"Found {len(buy_stocks)} Buy recommendations")
                    create_detailed_stock_analysis(buy_stocks)
                else:
                    st.info("No Buy recommendations at this time.")
            
            with rec_tabs[2]:
                st.success(f"Found {len(recommended_stocks)} total recommendations")
                create_detailed_stock_analysis(recommended_stocks)
                
        else:
            st.warning("No investment recommendations found in the data.")
    
    with tab5:
        st.markdown("### 🔍 Individual Stock Analysis")
        
        if not df.empty:
            selected_stock = st.selectbox("Select Stock", df['symbol'].tolist())
            
            if selected_stock:
                stock_data = df[df['symbol'] == selected_stock].iloc[0]
                
                # Get current price data
                current_price, price_change, change_pct = get_current_price(stock_data['symbol'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Basic Information")
                    st.write(f"**Company:** {stock_data.get('company_name', 'N/A')}")
                    
                    if current_price:
                        price_color = "price-positive" if price_change >= 0 else "price-negative"
                        price_html = f"""
                        <div class="price-card">
                            <strong>Current Price:</strong> <span class="{price_color}">{format_currency(current_price)}</span><br>
                            <strong>Change:</strong> <span class="{price_color}">{format_currency(price_change)} ({format_percentage(change_pct)})</span>
                        </div>
                        """
                        st.html(price_html)
                    else:
                        st.write(f"**Current Price:** {format_currency(stock_data.get('current_price', 0))}")
                    
                    st.write(f"**Target Price:** {format_currency(stock_data.get('target_price', 0))}")
                    st.write(f"**Market Cap:** ₹{stock_data.get('market_cap', 0):.0f} Cr")
                
                with col2:
                    st.markdown("#### 📊 Key Metrics")
                    st.write(f"**Value Score:** {stock_data.get('value_score', 0):.1f}")
                    
                    rec_style = get_recommendation_style(stock_data.get('investment_grade', 'N/A'))
                    if rec_style:
                        grade_html = f'**Investment Grade:** <span class="{rec_style}">{stock_data.get("investment_grade", "N/A")}</span>'
                        st.html(grade_html)
                    else:
                        st.write(f"**Investment Grade:** {stock_data.get('investment_grade', 'N/A')}")
                    
                    st.write(f"**PE Ratio:** {stock_data.get('pe_ratio', 'N/A')}")
                    st.write(f"**ROE:** {stock_data.get('roe', 'N/A')}")
                    
                # Price prediction chart
                create_price_prediction_chart(stock_data)
    
    with tab6:
        st.markdown("### 📋 Complete Data Table")
        
        # Search functionality
        search_term = st.text_input("🔍 Search stocks by symbol or company name")
        if search_term:
            mask = df['symbol'].str.contains(search_term, case=False, na=False)
            if 'company_name' in df.columns:
                mask |= df['company_name'].str.contains(search_term, case=False, na=False)
            df = df[mask]
        
        # Display dataframe
        st.dataframe(df, use_container_width=True, height=600)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv_data,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Modern footer
    show_modern_footer()

def show_modern_footer():
    """Display modern footer with system information"""
    footer_html = f"""
    <div style="
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border-primary);
        text-align: center;
    ">
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="color: var(--accent-primary);">🚀</span>
                <span>AI Stock Screener Pro</span>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="color: var(--success);">🤖</span>
                <span>Powered by OpenAI GPT</span>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="color: var(--info);">📊</span>
                <span>Real-time Analytics</span>
            </div>
        </div>
        <div style="
            color: var(--text-muted);
            font-size: 0.875rem;
        ">
            Built with ❤️ for intelligent investing • Version 3.0 Dark Edition
        </div>
    </div>
    """
    
    st.html(footer_html)

if __name__ == "__main__":
    main()
