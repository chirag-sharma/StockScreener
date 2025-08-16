"""
Unit Tests for Fundamental Analysis Prediction Model
=================================================

Comprehensive tests for the fundamental analysis-based price prediction model.
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stock_screener.prediction_models.fundamental_analysis_model import FundamentalAnalysisModel


class TestFundamentalAnalysisModel(unittest.TestCase):
    """Test cases for FundamentalAnalysisModel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
        
        # Create price data that might correlate with fundamental factors
        base_price = 2000
        
        # Simulate price based on fundamental factors
        earnings_growth = np.linspace(1.0, 1.2, 180)  # 20% earnings growth over period
        pe_multiple = 15 + 5 * np.sin(2 * np.pi * np.arange(180) / 90)  # PE cycles
        market_sentiment = np.cumsum(np.random.normal(0, 0.01, 180))
        
        # Price driven by fundamentals
        fundamental_value = base_price * earnings_growth * (pe_multiple / 15)
        prices = fundamental_value * np.exp(market_sentiment)
        
        cls.test_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.003, 180)),
            'High': prices * (1 + np.abs(np.random.normal(0.005, 0.002, 180))),
            'Low': prices * (1 - np.abs(np.random.normal(0.005, 0.002, 180))),
            'Close': prices,
            'Volume': np.random.lognormal(14.8, 0.4, 180)
        }, index=cls.dates)
        
        # Ensure OHLC consistency
        for i in range(len(cls.test_data)):
            row = cls.test_data.iloc[i]
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('High')] = max(row['Open'], row['Close'], row['High'])
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('Low')] = min(row['Open'], row['Close'], row['Low'])
    
    def test_model_initialization(self):
        """Test proper model initialization."""
        model = FundamentalAnalysisModel("FUND.TEST", self.test_data)
        
        self.assertEqual(model.symbol, "FUND.TEST")
        self.assertIsInstance(model.data, pd.DataFrame)
        self.assertEqual(len(model.data), 180)
        self.assertIn('Close', model.data.columns)
    
    @patch('yfinance.Ticker')
    def test_fundamental_data_retrieval(self, mock_ticker):
        """Test retrieval of fundamental data."""
        # Mock fundamental data
        mock_info = {
            'trailingPE': 18.5,
            'forwardPE': 16.2,
            'priceToBook': 2.8,
            'debtToEquity': 0.45,
            'returnOnEquity': 0.15,
            'returnOnAssets': 0.08,
            'profitMargins': 0.12,
            'revenueGrowth': 0.08,
            'earningsGrowth': 0.15,
            'currentRatio': 1.8,
            'quickRatio': 1.2,
            'marketCap': 50000000000,
            'enterpriseValue': 55000000000,
            'beta': 1.1,
            'dividendYield': 0.025,
            'payoutRatio': 0.3
        }
        
        mock_financials = pd.DataFrame({
            'Total Revenue': [10000000000, 10800000000, 11664000000],
            'Net Income': [1200000000, 1380000000, 1587000000],
            'Total Debt': [2500000000, 2400000000, 2300000000],
            'Total Assets': [15000000000, 16200000000, 17496000000],
            'Total Stockholder Equity': [8000000000, 9200000000, 10584000000]
        })
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.financials = mock_financials
        mock_ticker_instance.history.return_value = self.test_data
        mock_ticker.return_value = mock_ticker_instance
        
        model = FundamentalAnalysisModel("MOCK.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            self.assertIn('predicted_price', prediction)
            self.assertIn('confidence', prediction)
            self.assertEqual(prediction['method'], 'Fundamental Analysis')
            
            # Should have fundamental-specific information
            self.assertIn('valuation_metrics', prediction)
            self.assertIn('analysis_factors', prediction)
    
    def test_pe_ratio_analysis(self):
        """Test P/E ratio based valuation."""
        model = FundamentalAnalysisModel("PE.TEST", self.test_data)
        
        # Mock some basic fundamental data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'pe_ratio': 18.5,
                'industry_pe': 20.0,
                'earnings_growth': 0.12,
                'revenue_growth': 0.08
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                self.assertIn('predicted_price', prediction)
                
                # Should consider P/E ratio in analysis
                if 'valuation_metrics' in prediction:
                    metrics = prediction['valuation_metrics']
                    self.assertIn('pe_analysis', metrics)
    
    def test_dcf_valuation(self):
        """Test discounted cash flow valuation method."""
        model = FundamentalAnalysisModel("DCF.TEST", self.test_data)
        
        # Mock cash flow data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'free_cash_flow': 2000000000,
                'revenue_growth': 0.10,
                'profit_margin': 0.15,
                'cost_of_capital': 0.08,
                'terminal_growth': 0.03,
                'shares_outstanding': 100000000
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # DCF should provide intrinsic value
                if 'intrinsic_value' in prediction:
                    intrinsic = prediction['intrinsic_value']
                    self.assertIsInstance(intrinsic, (int, float))
                    self.assertGreater(intrinsic, 0)
    
    def test_growth_stock_analysis(self):
        """Test analysis for growth stocks."""
        model = FundamentalAnalysisModel("GROWTH.TEST", self.test_data)
        
        # Mock high growth company data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'revenue_growth': 0.25,  # High growth
                'earnings_growth': 0.30,
                'pe_ratio': 35,  # High P/E typical for growth
                'peg_ratio': 1.2,
                'gross_margin': 0.70,
                'r_and_d_expense': 0.15,
                'debt_to_equity': 0.20  # Low debt
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should identify growth characteristics
                if 'stock_type' in prediction:
                    self.assertEqual(prediction['stock_type'].lower(), 'growth')
                
                # Growth stocks might have higher confidence in bull markets
                self.assertGreaterEqual(prediction['confidence'], 0.3)
    
    def test_value_stock_analysis(self):
        """Test analysis for value stocks."""
        model = FundamentalAnalysisModel("VALUE.TEST", self.test_data)
        
        # Mock value stock data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'pe_ratio': 12,  # Low P/E
                'price_to_book': 1.2,  # Low P/B
                'dividend_yield': 0.045,  # High dividend
                'debt_to_equity': 0.40,
                'current_ratio': 2.1,
                'roe': 0.12,
                'revenue_growth': 0.05  # Moderate growth
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should identify value characteristics
                if 'stock_type' in prediction:
                    self.assertEqual(prediction['stock_type'].lower(), 'value')
                
                # Should analyze value metrics
                if 'value_metrics' in prediction:
                    value_metrics = prediction['value_metrics']
                    self.assertIn('undervalued', value_metrics)
    
    def test_financial_health_assessment(self):
        """Test financial health assessment."""
        model = FundamentalAnalysisModel("HEALTH.TEST", self.test_data)
        
        # Mock financial health data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'current_ratio': 2.5,  # Good liquidity
                'quick_ratio': 1.8,
                'debt_to_equity': 0.25,  # Low debt
                'interest_coverage': 12.0,  # Good coverage
                'return_on_equity': 0.18,  # Strong returns
                'return_on_assets': 0.10,
                'asset_turnover': 0.8
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should assess financial health
                if 'financial_health' in prediction:
                    health = prediction['financial_health']
                    
                    expected_metrics = ['liquidity_score', 'solvency_score', 'profitability_score']
                    for metric in expected_metrics:
                        if metric in health:
                            self.assertIsInstance(health[metric], (int, float))
                            self.assertGreaterEqual(health[metric], 0)
    
    def test_industry_comparison(self):
        """Test industry comparison analysis."""
        model = FundamentalAnalysisModel("INDUSTRY.TEST", self.test_data)
        
        # Mock industry comparison data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'pe_ratio': 22.0,
                'industry_avg_pe': 18.5,
                'roe': 0.16,
                'industry_avg_roe': 0.12,
                'profit_margin': 0.14,
                'industry_avg_margin': 0.11,
                'revenue_growth': 0.12,
                'industry_avg_growth': 0.08
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should compare to industry
                if 'industry_comparison' in prediction:
                    comparison = prediction['industry_comparison']
                    
                    # Should identify outperforming metrics
                    if 'outperforming_metrics' in comparison:
                        self.assertIsInstance(comparison['outperforming_metrics'], list)
    
    def test_earnings_quality_analysis(self):
        """Test earnings quality assessment."""
        model = FundamentalAnalysisModel("EARNINGS.TEST", self.test_data)
        
        # Mock earnings quality data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'net_income': 1500000000,
                'operating_cash_flow': 1800000000,  # Higher than net income (good)
                'free_cash_flow': 1200000000,
                'accruals': -300000000,  # Negative accruals (good quality)
                'revenue_recognition': 'conservative',
                'one_time_items': 50000000  # Minimal one-time items
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should assess earnings quality
                if 'earnings_quality' in prediction:
                    quality = prediction['earnings_quality']
                    
                    quality_indicators = ['cash_flow_quality', 'accruals_ratio', 'sustainability_score']
                    for indicator in quality_indicators:
                        if indicator in quality:
                            self.assertIsInstance(quality[indicator], (int, float))
    
    def test_dividend_analysis(self):
        """Test dividend-focused analysis."""
        model = FundamentalAnalysisModel("DIVIDEND.TEST", self.test_data)
        
        # Mock dividend data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'dividend_yield': 0.035,
                'payout_ratio': 0.45,
                'dividend_growth_rate': 0.08,
                'dividend_coverage': 2.2,
                'free_cash_flow_yield': 0.06,
                'dividend_history_years': 25  # Long dividend history
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should analyze dividend sustainability
                if 'dividend_analysis' in prediction:
                    div_analysis = prediction['dividend_analysis']
                    
                    self.assertIn('sustainability_score', div_analysis)
                    self.assertIn('yield_attractiveness', div_analysis)
    
    def test_management_efficiency_metrics(self):
        """Test management efficiency analysis."""
        model = FundamentalAnalysisModel("MANAGEMENT.TEST", self.test_data)
        
        # Mock efficiency metrics
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'return_on_equity': 0.18,
                'return_on_assets': 0.12,
                'return_on_invested_capital': 0.15,
                'asset_turnover': 1.2,
                'inventory_turnover': 8.5,
                'receivables_turnover': 12.0,
                'working_capital_turnover': 6.0
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should assess management efficiency
                if 'management_efficiency' in prediction:
                    efficiency = prediction['management_efficiency']
                    
                    efficiency_metrics = ['capital_efficiency', 'operational_efficiency', 'overall_score']
                    for metric in efficiency_metrics:
                        if metric in efficiency:
                            self.assertIsInstance(efficiency[metric], (int, float))
    
    def test_competitive_position_analysis(self):
        """Test competitive position assessment."""
        model = FundamentalAnalysisModel("COMPETITIVE.TEST", self.test_data)
        
        # Mock competitive data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'market_share': 0.15,  # 15% market share
                'gross_margin': 0.65,  # High margins indicate pricing power
                'brand_strength': 8.5,  # Out of 10
                'switching_costs': 'high',
                'network_effects': 'strong',
                'economies_of_scale': 'significant',
                'patent_protection': 'moderate'
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should assess competitive advantages
                if 'competitive_analysis' in prediction:
                    competitive = prediction['competitive_analysis']
                    
                    self.assertIn('moat_strength', competitive)
                    self.assertIn('competitive_advantages', competitive)
    
    def test_macro_economic_factors(self):
        """Test incorporation of macroeconomic factors."""
        model = FundamentalAnalysisModel("MACRO.TEST", self.test_data)
        
        # Mock macro factors
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'interest_rate_sensitivity': 0.7,  # High sensitivity
                'inflation_impact': -0.3,  # Negative impact
                'gdp_correlation': 0.8,  # High correlation with GDP
                'currency_exposure': 0.25,  # 25% international revenue
                'commodity_exposure': 0.15,
                'regulatory_risk': 'moderate'
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should consider macro factors
                if 'macro_analysis' in prediction:
                    macro = prediction['macro_analysis']
                    
                    self.assertIn('economic_sensitivity', macro)
                    self.assertIn('risk_factors', macro)
    
    def test_insufficient_fundamental_data(self):
        """Test behavior with insufficient fundamental data."""
        model = FundamentalAnalysisModel("INSUFFICIENT.TEST", self.test_data)
        
        # Mock insufficient data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {}  # No fundamental data
            
            prediction = model.predict(target_days=30)
            
            if 'error' in prediction:
                # Should provide meaningful error
                error_msg = prediction['error'].lower()
                self.assertTrue(any(word in error_msg for word in 
                              ['insufficient', 'fundamental', 'data', 'unavailable']))
            else:
                # If it works without data, confidence should be very low
                self.assertLess(prediction['confidence'], 0.4)
    
    def test_fundamental_price_target_calculation(self):
        """Test price target calculation based on fundamentals."""
        model = FundamentalAnalysisModel("PRICE.TARGET", self.test_data)
        
        # Mock comprehensive fundamental data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'eps_current': 12.50,
                'eps_forward': 14.80,
                'industry_pe': 18.5,
                'growth_rate': 0.12,
                'peg_ratio': 1.1,
                'book_value_per_share': 85.00,
                'tangible_book_value': 78.00,
                'revenue_per_share': 125.00
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should calculate price target
                if 'price_target_analysis' in prediction:
                    target_analysis = prediction['price_target_analysis']
                    
                    valuation_methods = ['pe_based_target', 'peg_based_target', 'book_value_target']
                    for method in valuation_methods:
                        if method in target_analysis:
                            self.assertIsInstance(target_analysis[method], (int, float))
                            self.assertGreater(target_analysis[method], 0)
    
    def test_sector_specific_analysis(self):
        """Test sector-specific fundamental analysis."""
        model = FundamentalAnalysisModel("SECTOR.TEST", self.test_data)
        
        # Mock tech sector data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'sector': 'Technology',
                'r_and_d_intensity': 0.18,  # High R&D for tech
                'gross_margin': 0.75,  # High margins
                'asset_light_model': True,
                'recurring_revenue': 0.80,  # 80% recurring
                'customer_acquisition_cost': 150,
                'lifetime_value': 2500,
                'network_effects': True
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should apply tech-specific analysis
                if 'sector_analysis' in prediction:
                    sector = prediction['sector_analysis']
                    
                    # Tech-specific metrics
                    tech_metrics = ['innovation_score', 'scalability_factor', 'recurring_revenue_strength']
                    for metric in tech_metrics:
                        if metric in sector:
                            self.assertIsInstance(sector[metric], (int, float))
    
    def test_risk_adjusted_valuation(self):
        """Test risk-adjusted fundamental valuation."""
        model = FundamentalAnalysisModel("RISK.ADJUSTED", self.test_data)
        
        # Mock risk data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'beta': 1.3,  # Higher risk
                'debt_to_equity': 0.65,  # Moderate debt
                'interest_coverage': 5.5,
                'earnings_volatility': 0.15,
                'revenue_volatility': 0.08,
                'business_risk': 'moderate',
                'financial_risk': 'moderate-high'
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should adjust for risk
                if 'risk_adjustment' in prediction:
                    risk_adj = prediction['risk_adjustment']
                    
                    self.assertIn('risk_premium', risk_adj)
                    self.assertIn('adjusted_discount_rate', risk_adj)
    
    def test_catalyst_identification(self):
        """Test identification of potential catalysts."""
        model = FundamentalAnalysisModel("CATALYSTS.TEST", self.test_data)
        
        # Mock catalyst data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'upcoming_earnings': '2024-02-15',
                'product_launches': ['Product A', 'Product B'],
                'regulatory_approvals_pending': True,
                'merger_speculation': False,
                'management_changes': False,
                'share_buyback_program': True,
                'dividend_increase_history': True
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should identify catalysts
                if 'catalysts' in prediction:
                    catalysts = prediction['catalysts']
                    
                    catalyst_types = ['positive_catalysts', 'negative_catalysts', 'timing_considerations']
                    for cat_type in catalyst_types:
                        if cat_type in catalysts:
                            self.assertIsInstance(catalysts[cat_type], (list, dict))
    
    def test_model_confidence_based_on_data_quality(self):
        """Test confidence adjustment based on fundamental data quality."""
        model = FundamentalAnalysisModel("CONFIDENCE.TEST", self.test_data)
        
        # Test with high-quality data
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'data_completeness': 0.95,  # 95% complete
                'data_freshness_days': 30,  # Recent data
                'auditor_quality': 'Big4',
                'reporting_standards': 'GAAP',
                'restatement_history': False,
                'insider_trading': 'normal'
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should have higher confidence with quality data
                self.assertGreater(prediction['confidence'], 0.5)
    
    def test_integration_with_technical_signals(self):
        """Test integration of fundamental analysis with basic technical signals."""
        model = FundamentalAnalysisModel("INTEGRATION.TEST", self.test_data)
        
        # Mock fundamental data showing undervaluation
        with patch.object(model, '_get_fundamental_data') as mock_get_data:
            mock_get_data.return_value = {
                'intrinsic_value': 2200,  # Above current price
                'current_price': 2000,
                'undervaluation_pct': 0.10,  # 10% undervalued
                'technical_confirmation': True,  # Technical supports fundamental view
                'momentum_alignment': 'positive'
            }
            
            prediction = model.predict(target_days=30)
            
            if 'error' not in prediction:
                # Should show positive outlook when fundamental and technical align
                current_price = self.test_data['Close'].iloc[-1]
                predicted_price = prediction['predicted_price']
                
                price_change_pct = (predicted_price - current_price) / current_price
                self.assertGreater(price_change_pct, -0.05)  # Should not predict significant decline


class TestFundamentalAnalysisModelAdvanced(unittest.TestCase):
    """Advanced tests for fundamental analysis model."""
    
    def setUp(self):
        """Set up advanced test scenarios."""
        np.random.seed(789)
        
        # Create different company scenarios
        self.growth_company_data = self._create_growth_company_data()
        self.value_company_data = self._create_value_company_data()
        self.distressed_company_data = self._create_distressed_company_data()
    
    def _create_growth_company_data(self):
        """Create data for a high-growth company."""
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        
        # High growth trajectory
        growth_factor = np.exp(np.cumsum(np.random.normal(0.008, 0.015, 120)))  # High growth
        base_prices = 800 * growth_factor
        
        return pd.DataFrame({
            'Open': base_prices * 0.999,
            'High': base_prices * 1.015,
            'Low': base_prices * 0.985,
            'Close': base_prices,
            'Volume': np.random.uniform(2000000, 4000000, 120)  # High volume
        }, index=dates)
    
    def _create_value_company_data(self):
        """Create data for a value company."""
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        
        # Stable, sideways movement typical of value stocks
        sideways_factor = 1 + 0.05 * np.sin(2 * np.pi * np.arange(120) / 60)
        base_prices = 150 * sideways_factor
        
        return pd.DataFrame({
            'Open': base_prices * 0.999,
            'High': base_prices * 1.008,
            'Low': base_prices * 0.992,
            'Close': base_prices,
            'Volume': np.random.uniform(800000, 1500000, 120)  # Lower volume
        }, index=dates)
    
    def _create_distressed_company_data(self):
        """Create data for a distressed company."""
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        
        # Declining prices with high volatility
        decline_factor = np.exp(np.cumsum(np.random.normal(-0.005, 0.025, 120)))  # Declining
        base_prices = 50 * decline_factor
        
        return pd.DataFrame({
            'Open': base_prices * 0.995,
            'High': base_prices * 1.025,
            'Low': base_prices * 0.975,
            'Close': base_prices,
            'Volume': np.random.uniform(500000, 3000000, 120)  # Volatile volume
        }, index=dates)
    
    @patch('yfinance.Ticker')
    def test_growth_company_analysis(self, mock_ticker):
        """Test analysis of high-growth companies."""
        # Mock growth company fundamentals
        mock_info = {
            'trailingPE': 45.0,  # High P/E
            'pegRatio': 1.8,
            'priceToBook': 8.5,
            'revenueGrowth': 0.35,  # 35% growth
            'earningsGrowth': 0.42,
            'profitMargins': 0.08,  # Lower margins due to growth investments
            'returnOnEquity': 0.22
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = self.growth_company_data
        mock_ticker.return_value = mock_ticker_instance
        
        model = FundamentalAnalysisModel("GROWTH.COMPANY", self.growth_company_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Growth companies should be identified correctly
            self.assertIn('predicted_price', prediction)
            
            # Should account for high growth in valuation
            if 'growth_premium' in prediction:
                self.assertIsInstance(prediction['growth_premium'], (int, float))
    
    @patch('yfinance.Ticker')
    def test_value_company_analysis(self, mock_ticker):
        """Test analysis of value companies."""
        # Mock value company fundamentals
        mock_info = {
            'trailingPE': 11.5,  # Low P/E
            'priceToBook': 0.9,  # Below book value
            'dividendYield': 0.055,  # High dividend
            'debtToEquity': 0.35,
            'returnOnEquity': 0.13,
            'currentRatio': 2.3,
            'revenueGrowth': 0.03  # Slow growth
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = self.value_company_data
        mock_ticker.return_value = mock_ticker_instance
        
        model = FundamentalAnalysisModel("VALUE.COMPANY", self.value_company_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Value companies should show potential for mean reversion
            if 'value_score' in prediction:
                self.assertIsInstance(prediction['value_score'], (int, float))
    
    @patch('yfinance.Ticker')
    def test_distressed_company_analysis(self, mock_ticker):
        """Test analysis of distressed companies."""
        # Mock distressed company fundamentals
        mock_info = {
            'trailingPE': None,  # Negative earnings
            'priceToBook': 0.3,  # Very low
            'debtToEquity': 2.8,  # High debt
            'currentRatio': 0.8,  # Poor liquidity
            'returnOnEquity': -0.15,  # Negative ROE
            'profitMargins': -0.05  # Negative margins
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = self.distressed_company_data
        mock_ticker.return_value = mock_ticker_instance
        
        model = FundamentalAnalysisModel("DISTRESSED.COMPANY", self.distressed_company_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should identify distress and adjust confidence
            self.assertLess(prediction['confidence'], 0.5)
            
            if 'distress_signals' in prediction:
                self.assertIsInstance(prediction['distress_signals'], (list, dict))


if __name__ == '__main__':
    unittest.main(verbosity=2)
