"""
Streamlit dashboard for visualizing and filtering stock screener results.
"""

import streamlit as st
import pandas as pd

# Load data from Excel file
@st.cache_data
def load_data():
    """
    Loads stock screener results from an Excel file.
    Returns:
        pd.DataFrame: DataFrame containing screener results.
    """
    return pd.read_excel("nifty50_screener_results.xlsx")

df = load_data()

# UI Title
st.title("Nifty 50 Stock Screener Dashboard")

# Sidebar filter options
st.sidebar.header("Filter Options")

roe = st.sidebar.slider("ROE (%)", 0, 50, (15, 50))
pe = st.sidebar.slider("P/E Ratio", 0, 50, (0, 25))
rsi = st.sidebar.slider("RSI (14)", 0, 100, (40, 70))

# Filter data based on user selections
filtered_df = df[
    (df["ROE (%)"] >= roe[0]) & (df["ROE (%)"] <= roe[1]) &
    (df["PE Ratio"] >= pe[0]) & (df["PE Ratio"] <= pe[1]) &
    (df["RSI (14)"] >= rsi[0]) & (df["RSI (14)"] <= rsi[1])
]

# Display filtered results
st.subheader(f"Filtered Stocks ({len(filtered_df)} found)")
st.dataframe(filtered_df)
