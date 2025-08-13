"""
Professional Streamlit dashboard for comprehensive stock analysis visualization.
Integrates AI analysis, multi-period predictions, and value investing insights.
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

# Configure page
st.set_page_config(
    page_title="AI-Powered Stock Screener Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .investment-grade-strong-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .investment-grade-buy {
        background-color: #cce5ff;
        color: #004085;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .investment-grade-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_comprehensive_data():
    """
    Loads the latest comprehensive analysis results from Excel file.
    Returns:
        pd.DataFrame: Complete analysis dataset with AI and predictions
    """
    try:
        # Look for the latest comprehensive analysis file
        output_dir = Path("data/output")
        
        # Find comprehensive analysis files with different patterns
        analysis_files = []
        analysis_files.extend(list(output_dir.glob("comprehensive_analysis_*.xlsx")))
        analysis_files.extend(list(output_dir.glob("comprehensive_analysis.xlsx")))
        
        if not analysis_files:
            st.error("❌ No comprehensive analysis files found in data/output/")
            st.error("📋 Please run: python scripts/run_screener.py")
            return None
            
        # Get the most recent file
        latest_file = max(analysis_files, key=os.path.getctime)
        
        st.success(f"📊 Loading data from: {latest_file.name}")
        
        # Load the Excel file
        df = pd.read_excel(latest_file)
        
        # Clean and prepare data
        df = clean_and_prepare_data(df)
        
        return df, latest_file.name
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def clean_and_prepare_data(df):
    """
    Clean and prepare the dataframe for dashboard use.
    """
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Clean string columns
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].fillna('N/A')
    
    # Ensure proper data types
    if 'PE Ratio' in df.columns:
        df['PE Ratio'] = pd.to_numeric(df['PE Ratio'], errors='coerce').fillna(0)
    if 'ROE' in df.columns:
        df['ROE'] = pd.to_numeric(df['ROE'], errors='coerce').fillna(0)
    if 'Value Score (1-10)' in df.columns:
        df['Value Score (1-10)'] = pd.to_numeric(df['Value Score (1-10)'], errors='coerce').fillna(0)
    
    return df

def create_investment_grade_chart(df):
    """Create investment grade distribution chart"""
    if 'Investment Recommendation' in df.columns:
        grade_counts = df['Investment Recommendation'].value_counts()
        
        # Dynamic color mapping based on actual values
        color_map = {
            'Strong Buy': '#28a745',
            'Buy': '#17a2b8', 
            'Hold': '#ffc107',
            'Weak Hold': '#fd7e14',
            'Sell': '#dc3545',
            'Strong Sell': '#6c757d',
            'Avoid': '#dc3545'
        }
        
        # Use default colors for any unmapped values
        colors = [color_map.get(grade, '#6c757d') for grade in grade_counts.index]
        
        fig = px.pie(
            values=grade_counts.values, 
            names=grade_counts.index,
            title="Investment Grade Distribution",
            color_discrete_sequence=colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    return None

def create_ai_score_distribution(df):
    """Create AI value score distribution"""
    if 'Value Score (1-10)' in df.columns:
        fig = px.histogram(
            df, 
            x='Value Score (1-10)',
            title="AI Value Score Distribution",
            nbins=10,
            labels={'Value Score (1-10)': 'AI Value Score', 'count': 'Number of Stocks'}
        )
        
        fig.update_layout(height=400)
        fig.update_traces(marker_color='skyblue', marker_line_color='navy', marker_line_width=1)
        
        return fig
    return None

def create_prediction_comparison(df):
    """Create multi-period prediction comparison"""
    prediction_cols = [col for col in df.columns if 'Price Target' in col or 'Growth' in col]
    
    if len(prediction_cols) >= 2:
        # Use first 20 stocks for better readability
        sample_df = df.head(20)
        
        fig = go.Figure()
        
        if 'Growth 6M (%)' in df.columns:
            fig.add_trace(go.Bar(
                name='6-Month Growth %',
                x=sample_df['Symbol'] if 'Symbol' in sample_df.columns else sample_df.index,
                y=sample_df['Growth 6M (%)'],
                marker_color='lightblue'
            ))
            
        if 'Growth 12M (%)' in df.columns:
            fig.add_trace(go.Bar(
                name='12-Month Growth %',
                x=sample_df['Symbol'] if 'Symbol' in sample_df.columns else sample_df.index,
                y=sample_df['Growth 12M (%)'],
                marker_color='darkblue'
            ))
        
        fig.update_layout(
            title="Multi-Period Growth Predictions (Top 20 Stocks)",
            xaxis_title="Stock Symbol",
            yaxis_title="Expected Growth (%)",
            barmode='group',
            height=400
        )
        
        return fig
    return None

def create_value_metrics_scatter(df):
    """Create PE vs ROE scatter plot with AI scores"""
    if all(col in df.columns for col in ['PE Ratio', 'ROE', 'Value Score (1-10)']):
        fig = px.scatter(
            df,
            x='PE Ratio',
            y='ROE',
            size='Value Score (1-10)',
            color='Investment Recommendation' if 'Investment Recommendation' in df.columns else 'Value Score (1-10)',
            hover_name='Symbol' if 'Symbol' in df.columns else 'Company Name',
            title="Value Analysis: PE Ratio vs ROE (Bubble size = AI Score)",
            labels={'PE Ratio': 'PE Ratio', 'ROE': 'Return on Equity (%)'}
        )
        
        fig.update_layout(height=500)
        
        return fig
    return None

# Main dashboard layout
def main():
    # Load data
    data_result = load_comprehensive_data()
    
    if data_result is None or data_result[0] is None:
        st.stop()
    
    df, filename = data_result
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI-Powered Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Stocks Analyzed", len(df))
    
    with col2:
        if 'Value Score (1-10)' in df.columns:
            avg_score = df['Value Score (1-10)'].mean()
            st.metric("🧠 Average AI Score", f"{avg_score:.1f}/10")
    
    with col3:
        if 'Investment Recommendation' in df.columns:
            # Count positive recommendations (Buy, Strong Buy, Hold)
            positive_recs = df[df['Investment Recommendation'].isin(['Strong Buy', 'Buy', 'Hold'])]['Investment Recommendation'].count()
            st.metric("⭐ Positive Recommendations", positive_recs)
    
    with col4:
        if 'Prediction Confidence' in df.columns:
            avg_confidence = df['Prediction Confidence'].mean()
            st.metric("🔮 Avg Prediction Confidence", f"{avg_confidence:.1f}%")
    
    # Sidebar filters
    st.sidebar.header("🔍 Advanced Filtering")
    
    # AI Score filter
    if 'Value Score (1-10)' in df.columns:
        ai_score_range = st.sidebar.slider(
            "AI Value Score", 
            float(df['Value Score (1-10)'].min()), 
            float(df['Value Score (1-10)'].max()), 
            (7.0, 10.0)
        )
    
    # Investment grade filter
    if 'Investment Recommendation' in df.columns:
        available_grades = df['Investment Recommendation'].unique()
        # Set default to actual available values that are positive
        default_grades = []
        for grade in available_grades:
            if grade in ['Strong Buy', 'Buy', 'Hold']:
                default_grades.append(grade)
        
        # If no positive grades found, include all available grades
        if not default_grades:
            default_grades = list(available_grades)
        
        selected_grades = st.sidebar.multiselect(
            "Investment Recommendations",
            available_grades,
            default=default_grades[:2] if len(default_grades) >= 2 else default_grades
        )
    
    # Financial metrics filters
    st.sidebar.subheader("📈 Financial Metrics")
    
    if 'PE Ratio' in df.columns:
        pe_range = st.sidebar.slider(
            "PE Ratio", 
            0.0, 
            float(df['PE Ratio'].max()), 
            (0.0, 25.0)
        )
    
    if 'ROE' in df.columns:
        roe_range = st.sidebar.slider(
            "ROE (%)", 
            float(df['ROE'].min()), 
            float(df['ROE'].max()), 
            (15.0, 50.0)
        )
    
    if 'Debt/Equity' in df.columns:
        de_max = st.sidebar.slider(
            "Max Debt/Equity", 
            0.0, 
            2.0, 
            1.0
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'Value Score (1-10)' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['Value Score (1-10)'] >= ai_score_range[0]) & 
            (filtered_df['Value Score (1-10)'] <= ai_score_range[1])
        ]
    
    if 'Investment Recommendation' in df.columns and selected_grades:
        filtered_df = filtered_df[filtered_df['Investment Recommendation'].isin(selected_grades)]
    
    if 'PE Ratio' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['PE Ratio'] >= pe_range[0]) & 
            (filtered_df['PE Ratio'] <= pe_range[1])
        ]
    
    if 'ROE' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['ROE'] >= roe_range[0]) & 
            (filtered_df['ROE'] <= roe_range[1])
        ]
    
    if 'Debt/Equity' in df.columns:
        filtered_df = filtered_df[filtered_df['Debt/Equity'] <= de_max]
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "💡 AI Analysis", "📋 Detailed Data"])
    
    with tab1:
        st.subheader(f"📈 Market Overview ({len(filtered_df)} stocks match criteria)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Investment grade distribution
            grade_chart = create_investment_grade_chart(filtered_df)
            if grade_chart:
                st.plotly_chart(grade_chart, use_container_width=True)
        
        with col2:
            # AI score distribution
            score_chart = create_ai_score_distribution(filtered_df)
            if score_chart:
                st.plotly_chart(score_chart, use_container_width=True)
        
        # Value metrics scatter plot
        scatter_chart = create_value_metrics_scatter(filtered_df)
        if scatter_chart:
            st.plotly_chart(scatter_chart, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Multi-Period Price Predictions")
        
        # Prediction comparison chart
        prediction_chart = create_prediction_comparison(filtered_df)
        if prediction_chart:
            st.plotly_chart(prediction_chart, use_container_width=True)
        
        # Top predictions table
        prediction_cols = ['Symbol', 'Company Name', 'Price Target (6M)', 'Price Target (12M)', 
                          'Growth 6M (%)', 'Growth 12M (%)', 'Prediction Confidence']
        available_pred_cols = [col for col in prediction_cols if col in filtered_df.columns]
        
        if available_pred_cols:
            st.subheader("🎯 Top Price Prediction Opportunities")
            top_predictions = filtered_df.nlargest(10, 'Growth 12M (%)' if 'Growth 12M (%)' in filtered_df.columns else available_pred_cols[-1])
            st.dataframe(top_predictions[available_pred_cols], use_container_width=True)
    
    with tab3:
        st.subheader("🧠 AI-Powered Analysis Insights")
        
        # Top AI-rated stocks
        if 'Value Score (1-10)' in filtered_df.columns:
            ai_cols = ['Symbol', 'Company Name', 'Value Score (1-10)', 'Investment Recommendation', 
                      'AI Sentiment', 'Financial Health', 'Risk Level']
            available_ai_cols = [col for col in ai_cols if col in filtered_df.columns]
            
            if available_ai_cols:
                st.subheader("⭐ Top AI-Rated Investment Opportunities")
                top_ai_stocks = filtered_df.nlargest(10, 'Value Score (1-10)')
                st.dataframe(top_ai_stocks[available_ai_cols], use_container_width=True)
        
        # AI reasoning for top stocks
        if 'AI Reasoning' in filtered_df.columns:
            st.subheader("💭 AI Investment Reasoning (Top 3 Stocks)")
            for i, (_, stock) in enumerate(filtered_df.head(3).iterrows()):
                with st.expander(f"📈 {stock.get('Symbol', f'Stock {i+1}')} - {stock.get('Company Name', 'N/A')}"):
                    st.write(f"**AI Score:** {stock.get('Value Score (1-10)', 'N/A')}/10")
                    st.write(f"**Recommendation:** {stock.get('Investment Recommendation', 'N/A')}")
                    st.write(f"**Reasoning:** {stock.get('AI Reasoning', 'N/A')}")
    
    with tab4:
        st.subheader("📋 Complete Filtered Dataset")
        st.info(f"Showing {len(filtered_df)} stocks out of {len(df)} total stocks analyzed")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search by symbol or company name:")
        with col2:
            show_all_cols = st.checkbox("Show all columns", value=False)
        
        # Apply search filter
        if search_term:
            search_mask = (
                filtered_df['Symbol'].str.contains(search_term, case=False, na=False) |
                filtered_df['Company Name'].str.contains(search_term, case=False, na=False)
            ) if 'Company Name' in filtered_df.columns else filtered_df['Symbol'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[search_mask]
        
        # Select columns to display
        if show_all_cols:
            display_df = filtered_df
        else:
            # Show key columns
            key_columns = ['Symbol', 'Company Name', 'PE Ratio', 'ROE', 'Value Score (1-10)', 
                          'Investment Recommendation', 'Target Price', 'Growth 12M (%)', 'Prediction Confidence']
            available_key_cols = [col for col in key_columns if col in filtered_df.columns]
            display_df = filtered_df[available_key_cols] if available_key_cols else filtered_df
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data as CSV",
            csv,
            f"filtered_stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key='download-csv'
        )
    
    # Footer information
    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 Data source: {filename}")
    st.sidebar.info(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
