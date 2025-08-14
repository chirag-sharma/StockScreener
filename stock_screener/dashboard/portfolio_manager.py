"""
Advanced Portfolio Management Module for Enhanced Dashboard
=========================================================

Features:
- Portfolio construction and optimization
- Risk metrics calculation
- Performance attribution analysis
- Sector allocation optimization
- Real-time portfolio tracking
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf

class PortfolioManager:
    """Advanced portfolio management and optimization"""
    
    def __init__(self, stocks_data: pd.DataFrame):
        self.stocks_data = stocks_data
        self.portfolio = {}
        self.risk_free_rate = 0.06  # 6% risk-free rate (typical for India)
    
    def create_optimal_portfolio(self, investment_amount: float = 1000000, 
                               max_positions: int = 10) -> Dict:
        """Create an optimal portfolio based on AI scores and risk metrics"""
        
        # Filter for STRONG_BUY and BUY grades
        eligible_stocks = self.stocks_data[
            self.stocks_data['investment_grade'].isin(['STRONG_BUY', 'BUY'])
        ].copy()
        
        if len(eligible_stocks) == 0:
            return {"error": "No eligible stocks found for portfolio construction"}
        
        # Sort by final score and select top stocks
        top_stocks = eligible_stocks.nlargest(max_positions, 'final_score')
        
        # Calculate weights based on normalized scores
        score_weights = top_stocks['final_score'] / top_stocks['final_score'].sum()
        
        # Apply position sizing limits (max 15% per position)
        max_weight_per_position = 0.15
        weights = np.minimum(score_weights, max_weight_per_position)
        weights = weights / weights.sum()  # Renormalize
        
        # Calculate investment amounts
        allocations = weights * investment_amount
        shares = allocations / top_stocks['current_price']
        
        portfolio = {
            'stocks': top_stocks[['symbol', 'current_price', 'target_price', 
                                'final_score', 'investment_grade']].to_dict('records'),
            'weights': weights.tolist(),
            'allocations': allocations.tolist(),
            'shares': shares.astype(int).tolist(),
            'total_investment': investment_amount,
            'expected_return': self._calculate_expected_return(top_stocks, weights),
            'risk_metrics': self._calculate_portfolio_risk(top_stocks, weights)
        }
        
        return portfolio
    
    def _calculate_expected_return(self, stocks: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate expected portfolio return based on target prices"""
        expected_returns = (stocks['target_price'] - stocks['current_price']) / stocks['current_price']
        portfolio_return = np.dot(weights, expected_returns)
        return portfolio_return * 100  # Convert to percentage
    
    def _calculate_portfolio_risk(self, stocks: pd.DataFrame, weights: np.ndarray) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            # Get historical price data for risk calculation
            price_data = {}
            for symbol in stocks['symbol']:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="1y")
                if not hist.empty:
                    price_data[symbol] = hist['Close'].pct_change().dropna()
            
            if not price_data:
                return {"error": "Unable to fetch historical data for risk calculation"}
            
            # Create returns dataframe
            returns_df = pd.DataFrame(price_data)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                return {"volatility": "N/A", "sharpe_ratio": "N/A", "var_95": "N/A"}
            
            # Calculate portfolio metrics
            portfolio_returns = returns_df.dot(weights)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Sharpe ratio
            excess_return = portfolio_returns.mean() * 252 - self.risk_free_rate
            sharpe_ratio = excess_return / (portfolio_returns.std() * np.sqrt(252))
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) * 100
            
            return {
                "volatility": f"{portfolio_volatility:.2f}%",
                "sharpe_ratio": f"{sharpe_ratio:.2f}",
                "var_95": f"{var_95:.2f}%"
            }
            
        except Exception as e:
            return {"error": f"Risk calculation failed: {str(e)}"}
    
    def create_portfolio_visualization(self, portfolio: Dict) -> go.Figure:
        """Create comprehensive portfolio visualization"""
        if 'error' in portfolio:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Allocation', 'Expected vs Current Value',
                          'Risk-Return Profile', 'Investment Grade Distribution'),
            specs=[[{"type": "domain"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "domain"}]]
        )
        
        # Portfolio allocation pie chart
        symbols = [stock['symbol'] for stock in portfolio['stocks']]
        allocations = portfolio['allocations']
        
        fig.add_trace(
            go.Pie(
                labels=symbols,
                values=allocations,
                name="Allocation",
                textinfo='label+percent',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Expected vs Current value
        current_values = [stock['current_price'] * shares for stock, shares 
                         in zip(portfolio['stocks'], portfolio['shares'])]
        target_values = [stock['target_price'] * shares for stock, shares 
                        in zip(portfolio['stocks'], portfolio['shares'])]
        
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=current_values,
                mode='markers+lines',
                name='Current Value',
                marker=dict(size=10, color='blue')
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=target_values,
                mode='markers+lines',
                name='Target Value',
                marker=dict(size=10, color='green')
            ),
            row=1, col=2
        )
        
        # Risk-Return scatter
        scores = [stock['final_score'] for stock in portfolio['stocks']]
        expected_returns = [(stock['target_price'] - stock['current_price']) / 
                           stock['current_price'] * 100 for stock in portfolio['stocks']]
        
        fig.add_trace(
            go.Scatter(
                x=expected_returns,
                y=scores,
                mode='markers+text',
                text=symbols,
                textposition="top center",
                marker=dict(size=15, color='red', opacity=0.7),
                name='Risk-Return'
            ),
            row=2, col=1
        )
        
        # Investment grade distribution
        grades = [stock['investment_grade'] for stock in portfolio['stocks']]
        grade_counts = pd.Series(grades).value_counts()
        
        colors = {'STRONG_BUY': '#2ecc71', 'BUY': '#3498db', 'HOLD': '#f39c12'}
        
        fig.add_trace(
            go.Pie(
                labels=grade_counts.index,
                values=grade_counts.values,
                name="Investment Grades",
                marker_colors=[colors.get(grade, '#95a5a6') for grade in grade_counts.index],
                textinfo='label+value'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="📊 Portfolio Analysis Dashboard",
            title_x=0.5
        )
        
        return fig
    
    def calculate_sector_allocation(self, portfolio: Dict) -> pd.DataFrame:
        """Calculate sector-wise allocation if sector data is available"""
        if 'error' in portfolio or 'sector' not in self.stocks_data.columns:
            return pd.DataFrame()
        
        portfolio_df = pd.DataFrame(portfolio['stocks'])
        portfolio_df['allocation'] = portfolio['allocations']
        
        # Merge with sector data
        sector_data = self.stocks_data[['symbol', 'sector']].set_index('symbol')
        portfolio_df = portfolio_df.set_index('symbol').join(sector_data, how='left')
        
        sector_allocation = portfolio_df.groupby('sector')['allocation'].sum().reset_index()
        sector_allocation['percentage'] = (sector_allocation['allocation'] / 
                                         portfolio['total_investment'] * 100).round(2)
        
        return sector_allocation.sort_values('allocation', ascending=False)

def render_portfolio_tab(df: pd.DataFrame):
    """Render the enhanced portfolio management tab"""
    st.markdown("### 💼 Advanced Portfolio Management")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(df)
    
    # Portfolio configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        investment_amount = st.number_input(
            "Investment Amount (₹)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            help="Total amount to invest"
        )
    
    with col2:
        max_positions = st.slider(
            "Max Positions",
            min_value=5,
            max_value=20,
            value=10,
            help="Maximum number of stocks in portfolio"
        )
    
    with col3:
        if st.button("🎯 Create Optimal Portfolio", type="primary"):
            st.session_state.portfolio_created = True
    
    # Create portfolio if button clicked
    if st.session_state.get('portfolio_created', False):
        with st.spinner("🔄 Creating optimal portfolio..."):
            portfolio = portfolio_manager.create_optimal_portfolio(
                investment_amount=investment_amount,
                max_positions=max_positions
            )
        
        if 'error' in portfolio:
            st.error(f"❌ {portfolio['error']}")
        else:
            # Portfolio summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{portfolio['expected_return']:.2f}%")
            
            with col2:
                risk_metrics = portfolio['risk_metrics']
                volatility = risk_metrics.get('volatility', 'N/A')
                st.metric("Portfolio Volatility", volatility)
            
            with col3:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 'N/A')
                st.metric("Sharpe Ratio", sharpe_ratio)
            
            with col4:
                var_95 = risk_metrics.get('var_95', 'N/A')
                st.metric("VaR (95%)", var_95)
            
            # Portfolio visualization
            fig = portfolio_manager.create_portfolio_visualization(portfolio)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio holdings table
            st.markdown("#### 📋 Portfolio Holdings")
            holdings_df = pd.DataFrame(portfolio['stocks'])
            holdings_df['Weight %'] = [f"{w*100:.1f}%" for w in portfolio['weights']]
            holdings_df['Allocation ₹'] = [f"₹{a:,.0f}" for a in portfolio['allocations']]
            holdings_df['Shares'] = portfolio['shares']
            holdings_df['Expected Return %'] = [
                f"{((row['target_price'] - row['current_price']) / row['current_price'] * 100):.1f}%" 
                for _, row in holdings_df.iterrows()
            ]
            
            # Display holdings
            st.dataframe(
                holdings_df[['symbol', 'current_price', 'target_price', 'final_score', 
                           'investment_grade', 'Weight %', 'Allocation ₹', 'Shares', 
                           'Expected Return %']],
                use_container_width=True,
                column_config={
                    'current_price': st.column_config.NumberColumn("Current ₹", format="%.2f"),
                    'target_price': st.column_config.NumberColumn("Target ₹", format="%.2f"),
                    'final_score': st.column_config.NumberColumn("Score", format="%.1f")
                }
            )
            
            # Sector allocation if available
            sector_allocation = portfolio_manager.calculate_sector_allocation(portfolio)
            if not sector_allocation.empty:
                st.markdown("#### 🏭 Sector Allocation")
                
                fig_sector = px.treemap(
                    sector_allocation,
                    path=['sector'],
                    values='allocation',
                    title='Portfolio Sector Allocation',
                    color='percentage',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            
            # Download portfolio
            portfolio_csv = holdings_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio (CSV)",
                data=portfolio_csv,
                file_name=f"optimal_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="secondary"
            )

def render_backtesting_tab(df: pd.DataFrame):
    """Render backtesting analysis tab"""
    st.markdown("### 📈 Strategy Backtesting")
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Strategy",
        ["AI Score Based", "Investment Grade Based", "Combined Score", "Sector Rotation"],
        help="Choose backtesting strategy"
    )
    
    # Backtesting parameters
    col1, col2 = st.columns(2)
    
    with col1:
        lookback_period = st.selectbox(
            "Lookback Period",
            ["1M", "3M", "6M", "1Y"],
            index=2,
            help="Historical period for backtesting"
        )
    
    with col2:
        rebalance_frequency = st.selectbox(
            "Rebalancing",
            ["Monthly", "Quarterly", "Semi-Annual"],
            index=1,
            help="Portfolio rebalancing frequency"
        )
    
    if st.button("🔄 Run Backtest", type="primary"):
        st.info("🚧 Backtesting functionality coming soon!")
        st.markdown("""
        **Planned Features:**
        - Historical performance simulation
        - Risk-adjusted returns analysis
        - Drawdown analysis
        - Benchmark comparison
        - Strategy optimization
        """)

# Add session state initialization
if 'portfolio_created' not in st.session_state:
    st.session_state.portfolio_created = False
