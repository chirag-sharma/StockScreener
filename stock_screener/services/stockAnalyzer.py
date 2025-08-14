#!/usr/bin/env python3
"""
Stock Analysis Service Module
=============================

This module provides the StockAnalyzer class for fetching and analyzing stock data 
using yfinance. It applies financial thresholds and returns structured analysis results
with comprehensive error handling and logging.

Features:
- Stock data fetching with yfinance integration
- Financial threshold evaluation
- Comprehensive error handling and validation
- Structured analysis result generation
- Performance metrics calculation
- Detailed logging for debugging

Usage:
    from stock_screener.services.stockAnalyzer import StockAnalyzer
    
    analyzer = StockAnalyzer('RELIANCE.NS')
    analyzer.fetch_data()
    result = analyzer.analyze()
"""

import yfinance as yf
import time
from datetime import datetime
from stock_screener.core.constants import THRESHOLDS
from stock_screener.utils.logging_config import get_logger, log_execution_start, log_execution_end

# Initialize module logger
logger = get_logger(__name__)


class StockAnalyzer:
    """
    Analyzes a stock's financial data and evaluates it against predefined thresholds.
    
    This class provides comprehensive stock analysis functionality including:
    - Data fetching from yfinance
    - Financial metric calculation
    - Threshold comparison
    - Result structuring and validation
    """
    
    def __init__(self, symbol):
        """
        Initialize the analyzer with a stock symbol.
        
        Args:
            symbol (str): The stock ticker symbol (e.g., 'RELIANCE.NS', 'TCS.NS')
        """
        self.symbol = symbol.strip().upper()
        self.info = {}
        self.analysis = {}
        self._fetch_timestamp = None
        
        logger.debug(f"StockAnalyzer initialized for symbol: {self.symbol}")
    
    def _validate_symbol(self):
        """
        Validate the stock symbol format.
        
        Returns:
            bool: True if symbol appears valid, False otherwise
        """
        if not self.symbol:
            logger.error("Empty symbol provided")
            return False
        
        if len(self.symbol) < 2:
            logger.error(f"Symbol too short: {self.symbol}")
            return False
        
        return True
    
    def fetch_data(self):
        """
        Fetch stock data using yfinance and store it in self.info.
        
        This method fetches comprehensive stock information including:
        - Basic company information
        - Financial metrics and ratios
        - Market data and pricing
        - Historical performance indicators
        
        Returns:
            bool: True if data was successfully fetched, False otherwise
        """
        if not self._validate_symbol():
            return False
        
        start_time = time.time()
        log_execution_start(__name__, "fetch_data", symbol=self.symbol)
        
        try:
            logger.info(f"Fetching data for symbol: {self.symbol}")
            
            stock = yf.Ticker(self.symbol)
            self.info = stock.info
            self._fetch_timestamp = datetime.now()
            
            if not self.info:
                logger.warning(f"No data returned for symbol: {self.symbol}")
                return False
            
            # Log key information about fetched data
            company_name = self.info.get('longName', 'Unknown')
            market_cap = self.info.get('marketCap', 'N/A')
            
            logger.info(f"Data fetched successfully - Company: {company_name}, Market Cap: {market_cap}")
            logger.debug(f"Data fields available: {len(self.info)} fields")
            
            duration = time.time() - start_time
            log_execution_end(__name__, "fetch_data", duration, f"Success: {company_name}")
            
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to fetch data for {self.symbol}: {e}")
            log_execution_end(__name__, "fetch_data", duration, f"Failed: {str(e)}")
            self.info = {}
            return False
    
    def _safe_get(self, key, transform=None, default=None):
        """
        Safely extract a value from stock info with optional transformation.
        
        Args:
            key (str): The key to extract from self.info
            transform (callable): Optional function to transform the value
            default: Default value if key is missing or transformation fails
            
        Returns:
            The extracted and optionally transformed value, or default
        """
        try:
            value = self.info.get(key, default)
            
            if value is None:
                return default
            
            if transform:
                return transform(value)
            
            return value
            
        except Exception as e:
            logger.debug(f"Error extracting/transforming {key}: {e}")
            return default
    
    def _calculate_derived_metrics(self):
        """
        Calculate derived financial metrics from base data.
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        derived = {}
        
        try:
            # Calculate additional ratios and metrics
            total_debt = self._safe_get('totalDebt', float, 0)
            total_equity = self._safe_get('totalStockholderEquity', float, 0)
            
            if total_equity and total_equity != 0:
                derived['debt_to_equity_calculated'] = total_debt / total_equity
            
            # Free cash flow yield calculation
            free_cash_flow = self._safe_get('freeCashflow', float, 0)
            market_cap = self._safe_get('marketCap', float, 0)
            
            if market_cap and market_cap != 0:
                derived['fcf_yield'] = (free_cash_flow / market_cap) * 100
            
            logger.debug(f"Calculated {len(derived)} derived metrics")
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
        
    
    def analyze(self):
        """
        Analyze the fetched stock data and return a comprehensive dictionary of metrics.
        
        This method performs complete financial analysis including:
        - Basic financial metrics calculation
        - Threshold comparison and pass/fail evaluation
        - Derived metrics calculation
        - Weighted value investing score
        - Investment recommendation generation
        
        Returns:
            dict: Comprehensive analysis results, or None if no data is available.
        """
        if not self.info:
            logger.warning(f"No data available for analysis of symbol: {self.symbol}")
            return None
        
        start_time = time.time()
        log_execution_start(__name__, "analyze", symbol=self.symbol)
        
        try:
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics()
            
            # Helper functions for complex calculations
            def calculate_price_to_cash_flow():
                """Calculate Price to Cash Flow ratio using Market Cap and Operating Cash Flow"""
                market_cap = self._safe_get('marketCap', float)
                revenue = self._safe_get('totalRevenue', float)
                op_margin = self._safe_get('operatingMargins', float)

                if all(v is not None for v in [market_cap, revenue, op_margin]):
                    operating_cash_flow = revenue * op_margin
                    if operating_cash_flow > 0:  # Avoid division by zero or negative values
                        return market_cap / operating_cash_flow
                return None

            def calculate_net_profit_margin():
                """Calculate Net Profit Margin with fallback options"""
                # First try using profitMargins directly
                profit_margins = self._safe_get('profitMargins', float)
                if profit_margins is not None:
                    return profit_margins * 100

                # Fallback to calculation from raw numbers
                net_income = self._safe_get('netIncomeToCommon', float)
                revenue = self._safe_get('totalRevenue', float)
                if net_income is not None and revenue is not None and revenue != 0:
                    return (net_income / revenue) * 100
                return None

            def calculate_interest_coverage():
                """Calculate Interest Coverage Ratio safely"""
                ebitda = self._safe_get('ebitda', float)
                interest_expense = self._safe_get('totalInterestExpense', float, 1)  # Default to 1 to avoid division by zero
                
                if ebitda is not None and interest_expense and interest_expense != 0:
                    return ebitda / interest_expense
                return None

            def calculate_cash_conversion():
                """Calculate Cash Conversion Ratio safely"""
                free_cash_flow = self._safe_get('freeCashflow', float)
                net_income = self._safe_get('netIncome', float, 1)  # Default to 1 to avoid division by zero
                
                if free_cash_flow is not None and net_income and net_income != 0:
                    return free_cash_flow / net_income
                return None

            # Collect key financial metrics
            self.analysis = {
                'Symbol': self.symbol,
                'Company Name': self._safe_get('longName', str, 'N/A'),
                'PE Ratio': self._safe_get('trailingPE', float),
                'Debt/Equity': self._safe_get('debtToEquity', float),
                'ROE': self._safe_get('returnOnEquity', lambda x: float(x) * 100),
                'Current Ratio': self._safe_get('currentRatio', float),
                'Price to Book': self._safe_get('priceToBook', float),
                'Promoter Holding': self._safe_get('heldPercentInsiders', lambda x: float(x) * 100),
                'Price to Cash Flow': calculate_price_to_cash_flow(),
                'Quick Ratio': self._safe_get('quickRatio', float),
                'Interest Coverage Ratio': calculate_interest_coverage(),
                'Free Cash Flow': self._safe_get('freeCashflow', float),
                'EPS Growth (%)': self._safe_get('earningsQuarterlyGrowth', lambda x: float(x) * 100),
                'Return on Assets (ROA)': self._safe_get('returnOnAssets', lambda x: float(x) * 100),
                'Net Profit Margin': calculate_net_profit_margin(),
                'Operating Margin': self._safe_get('operatingMargins', lambda x: float(x) * 100),
                'Cash Conversion Ratio': calculate_cash_conversion(),
                'Pledged Shares (%)': None,  # Not available in yfinance
                'EV/EBITDA': self._safe_get('enterpriseToEbitda', float),
                'Revenue Growth (%)': self._safe_get('revenueGrowth', lambda x: float(x) * 100),
                'Market Cap': self._safe_get('marketCap', float),
                'Sector': self._safe_get('sector', str, 'N/A'),
                'Industry': self._safe_get('industry', str, 'N/A'),
            }

            # Add derived metrics to analysis
            self.analysis.update(derived_metrics)

            # Log key calculations for debugging
            logger.debug(f"Key metrics calculated for {self.symbol}:")
            if self.analysis['Price to Cash Flow'] is not None:
                logger.debug(f"  Price to Cash Flow: {self.analysis['Price to Cash Flow']:.2f}")
            if self.analysis['Net Profit Margin'] is not None:
                logger.debug(f"  Net Profit Margin: {self.analysis['Net Profit Margin']:.2f}%")

            # Apply thresholds and generate pass/fail results
            self._evaluate_thresholds()
            
            # Calculate margin of safety
            self._calculate_margin_of_safety()
            
            # Calculate weighted value investing score
            self.analysis['Value Score'] = self._calculate_weighted_score()
            
            # Generate investment recommendation
            self._generate_recommendation()

            duration = time.time() - start_time
            score = self.analysis.get('Value Score', 0)
            recommendation = self.analysis.get('Investment Recommendation', 'Unknown')
            log_execution_end(__name__, "analyze", duration, f"Score: {score}, Rec: {recommendation}")

            return self.analysis
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Analysis failed for {self.symbol}: {e}")
            log_execution_end(__name__, "analyze", duration, f"Failed: {str(e)}")
            return None
    
    def _evaluate_thresholds(self):
        """Apply financial thresholds and generate pass/fail evaluations."""
        try:
            # Apply thresholds from constants
            threshold_evaluations = {
                'PE Ratio Pass': ('PE Ratio', '<', THRESHOLDS['pe_ratio_max']),
                'Debt/Equity Pass': ('Debt/Equity', '<', THRESHOLDS['debt_to_equity_max']),
                'ROE Pass': ('ROE', '>', THRESHOLDS['roe_min']),
                'Current Ratio Pass': ('Current Ratio', '>', THRESHOLDS['current_ratio_min']),
                'Price to Book Pass': ('Price to Book', '<', THRESHOLDS['price_to_book_max']),
                'Promoter Holding Pass': ('Promoter Holding', '>', THRESHOLDS['promoter_holding_min']),
                'Price to Cash Flow Pass': ('Price to Cash Flow', '<', THRESHOLDS['price_to_cash_flow_max']),
                'Quick Ratio Pass': ('Quick Ratio', '>', THRESHOLDS['quick_ratio_min']),
                'Interest Coverage Ratio Pass': ('Interest Coverage Ratio', '>', THRESHOLDS['interest_coverage_min']),
                'Free Cash Flow Pass': ('Free Cash Flow', '>', THRESHOLDS['free_cash_flow_min']),
                'EPS Growth (%) Pass': ('EPS Growth (%)', '>', THRESHOLDS['eps_growth_min']),
                'Return on Assets (ROA) Pass': ('Return on Assets (ROA)', '>', THRESHOLDS['roa_min']),
                'Net Profit Margin Pass': ('Net Profit Margin', '>', THRESHOLDS['net_profit_margin_min']),
                'Operating Margin Pass': ('Operating Margin', '>', THRESHOLDS['operating_margin_min']),
                'Cash Conversion Ratio Pass': ('Cash Conversion Ratio', '>', THRESHOLDS['cash_conversion_min']),
                'Pledged Shares Pass': ('Pledged Shares (%)', '<', THRESHOLDS['pledged_shares_max']),
                'EV/EBITDA Pass': ('EV/EBITDA', '<', THRESHOLDS['ev_ebitda_max']),
                'Revenue Growth (%) Pass': ('Revenue Growth (%)', '>', THRESHOLDS['revenue_growth_min']),
            }

            for pass_key, (metric_key, condition, threshold) in threshold_evaluations.items():
                self.analysis[pass_key] = self._check_threshold(metric_key, condition, threshold)
            
            # Count total passes for summary
            total_checks = len(threshold_evaluations)
            passed_checks = sum(1 for v in threshold_evaluations if self.analysis.get(f"{v} Pass", False))
            
            self.analysis['Threshold Summary'] = f"{passed_checks}/{total_checks} checks passed"
            logger.debug(f"Threshold evaluation completed: {self.analysis['Threshold Summary']}")
            
        except Exception as e:
            logger.error(f"Error evaluating thresholds: {e}")
    
    def _calculate_margin_of_safety(self):
        """Calculate margin of safety based on intrinsic value estimation."""
        try:
            free_cash_flow = self.analysis.get('Free Cash Flow')
            market_value = self.analysis.get('Market Cap')

            if free_cash_flow and market_value and market_value != 0:
                # Simple 10x FCF estimate for intrinsic value
                intrinsic_value = free_cash_flow * 10
                mos = (intrinsic_value - market_value) / market_value * 100
                self.analysis['Margin of Safety (%)'] = round(mos, 2)
                logger.debug(f"Margin of Safety calculated: {mos:.2f}%")
            else:
                self.analysis['Margin of Safety (%)'] = 'N/A'
                logger.debug("Insufficient data for Margin of Safety calculation")
                
        except Exception as e:
            logger.error(f"Error calculating margin of safety: {e}")
            self.analysis['Margin of Safety (%)'] = 'N/A'
    
    def _generate_recommendation(self):
        """Generate investment recommendation based on weighted score."""
        try:
            score = self.analysis.get('Value Score', 0)
            
            if score >= 90:
                recommendation = 'Strong Buy'
            elif score >= 70:
                recommendation = 'Buy'
            elif score >= 50:
                recommendation = 'Hold'
            elif score >= 30:
                recommendation = 'Weak Hold'
            else:
                recommendation = 'Avoid'
            
            self.analysis['Investment Recommendation'] = recommendation
            logger.debug(f"Investment recommendation generated: {recommendation} (Score: {score})")
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            self.analysis['Investment Recommendation'] = 'Unknown'

    def _calculate_weighted_score(self):
        """
        Calculate weighted value investing score based on fundamental principles.
        
        This method assigns weights to different financial metrics based on
        value investing principles and calculates a comprehensive score.
        
        Returns:
            float: Weighted score between 0-100 based on weighted criteria.
        """
        try:
            # Define weights for each metric based on value investing principles
            weights = {
                # Core Value Metrics (40% total)
                'PE Ratio': 15.0,
                'Price to Book': 10.0, 
                'EV/EBITDA': 10.0,
                'Margin of Safety (%)': 5.0,
                
                # Profitability & Quality (30% total)
                'ROE': 10.0,
                'Net Profit Margin': 8.0,
                'Operating Margin': 7.0,
                'Return on Assets (ROA)': 5.0,
                
                # Financial Strength (20% total)
                'Current Ratio': 6.0,
                'Debt/Equity': 6.0,
                'Quick Ratio': 4.0,
                'Interest Coverage Ratio': 4.0,
                
                # Growth & Cash Generation (10% total)
                'Free Cash Flow': 4.0,
                'EPS Growth (%)': 3.0,
                'Revenue Growth (%)': 3.0,
                
                # Additional Metrics (Remaining weight)
                'Price to Cash Flow': 2.5,
                'Promoter Holding': 2.0,
                'Cash Conversion Ratio': 1.5
            }
            
            total_score = 0.0
            total_applicable_weight = 0.0
            
            for metric, weight in weights.items():
                value = self.analysis.get(metric)
                if value is not None and value != 'N/A':
                    metric_score = self._score_metric(metric, value)
                    total_score += metric_score * weight
                    total_applicable_weight += weight
            
            # Calculate final weighted average score
            if total_applicable_weight > 0:
                final_score = total_score / total_applicable_weight
                logger.debug(f"Weighted score calculated: {final_score:.1f} (from {total_applicable_weight:.1f} total weight)")
                return round(final_score, 1)
            else:
                logger.warning("No applicable metrics for scoring")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return 0.0
    
    def _score_metric(self, metric, value):
        """
        Score individual metrics on a 0-100 scale based on value investing criteria.
        
        Args:
            metric (str): Name of the financial metric
            value: Value of the metric (numeric or string)
            
        Returns:
            float: Score between 0-100, higher scores indicate better value/quality
        """
        # Handle string values
        if isinstance(value, str):
            return 0.0
            
        try:
            value = float(value)
        except (ValueError, TypeError):
            logger.debug(f"Could not convert {metric} value to float: {value}")
            return 0.0
        
        # Scoring logic for each metric based on value investing principles
        if metric == 'PE Ratio':
            if value <= 0: return 0.0
            if value <= 10: return 100.0
            if value <= 15: return 80.0
            if value <= 20: return 60.0
            if value <= 25: return 40.0
            if value <= 30: return 20.0
            return 0.0
            
        elif metric == 'Price to Book':
            if value <= 0: return 0.0
            if value <= 1: return 100.0
            if value <= 1.5: return 80.0
            if value <= 2: return 60.0
            if value <= 3: return 40.0
            if value <= 4: return 20.0
            return 0.0
            
        elif metric == 'EV/EBITDA':
            if value <= 0: return 0.0
            if value <= 6: return 100.0
            if value <= 8: return 80.0
            if value <= 10: return 60.0
            if value <= 12: return 40.0
            if value <= 15: return 20.0
            return 0.0
            
        elif metric == 'Margin of Safety (%)':
            if value >= 30: return 100.0
            if value >= 20: return 80.0
            if value >= 10: return 60.0
            if value >= 0: return 40.0
            if value >= -10: return 20.0
            return 0.0
            
        elif metric in ['ROE', 'Return on Assets (ROA)', 'Net Profit Margin', 'Operating Margin']:
            if value >= 25: return 100.0
            if value >= 20: return 80.0
            if value >= 15: return 60.0
            if value >= 10: return 40.0
            if value >= 5: return 20.0
            return 0.0
            
        elif metric in ['Current Ratio', 'Quick Ratio']:
            if value >= 2.5: return 100.0
            if value >= 2.0: return 80.0
            if value >= 1.5: return 60.0
            if value >= 1.0: return 40.0
            if value >= 0.8: return 20.0
            return 0.0
            
        elif metric == 'Debt/Equity':
            if value <= 0.3: return 100.0
            if value <= 0.5: return 80.0
            if value <= 0.8: return 60.0
            if value <= 1.0: return 40.0
            if value <= 1.5: return 20.0
            return 0.0
            
        elif metric == 'Interest Coverage Ratio':
            if value >= 10: return 100.0
            if value >= 7: return 80.0
            if value >= 5: return 60.0
            if value >= 3: return 40.0
            if value >= 2: return 20.0
            return 0.0
            
        elif metric == 'Free Cash Flow':
            if value >= 1000000000: return 100.0  # 1B+
            if value >= 500000000: return 80.0    # 500M+
            if value >= 100000000: return 60.0    # 100M+
            if value >= 50000000: return 40.0     # 50M+
            if value >= 0: return 20.0            # Positive
            return 0.0
            
        elif metric in ['EPS Growth (%)', 'Revenue Growth (%)']:
            if value >= 25: return 100.0
            if value >= 15: return 80.0
            if value >= 10: return 60.0
            if value >= 5: return 40.0
            if value >= 0: return 20.0
            return 0.0
            
        elif metric == 'Price to Cash Flow':
            if value <= 0: return 0.0
            if value <= 8: return 100.0
            if value <= 12: return 80.0
            if value <= 15: return 60.0
            if value <= 20: return 40.0
            if value <= 25: return 20.0
            return 0.0
            
        elif metric == 'Promoter Holding':
            if value >= 70: return 100.0
            if value >= 60: return 80.0
            if value >= 50: return 60.0
            if value >= 40: return 40.0
            if value >= 25: return 20.0
            return 0.0
            
        elif metric == 'Cash Conversion Ratio':
            if value >= 1.5: return 100.0
            if value >= 1.2: return 80.0
            if value >= 1.0: return 60.0
            if value >= 0.8: return 40.0
            if value >= 0.5: return 20.0
            return 0.0
            
        # Default scoring for unknown metrics
        logger.debug(f"Using default scoring for unknown metric: {metric}")
        return 50.0

    def _check_threshold(self, key, condition, threshold):
        """
        Check if a metric passes a given threshold condition.
        
        Args:
            key (str): Metric key to check
            condition (str): Condition ('>' or '<')
            threshold (float): Threshold value
            
        Returns:
            bool: True if metric passes threshold, False otherwise
        """
        value = self.analysis.get(key)
        if value is None or value == 'N/A':
            return False
        
        try:
            value = float(value)
            if condition == '<':
                return value < threshold
            elif condition == '>':
                return value > threshold
            else:
                logger.error(f"Unknown condition: {condition}")
                return False
        except (ValueError, TypeError):
            logger.debug(f"Could not convert {key} value to float for threshold check: {value}")
            return False
    
    def get_analysis_summary(self):
        """
        Get a summary of the analysis results.
        
        Returns:
            dict: Summary of key analysis metrics
        """
        if not self.analysis:
            return {"error": "No analysis available"}
        
        return {
            'symbol': self.analysis.get('Symbol'),
            'company_name': self.analysis.get('Company Name'),
            'value_score': self.analysis.get('Value Score'),
            'recommendation': self.analysis.get('Investment Recommendation'),
            'pe_ratio': self.analysis.get('PE Ratio'),
            'roe': self.analysis.get('ROE'),
            'debt_equity': self.analysis.get('Debt/Equity'),
            'margin_of_safety': self.analysis.get('Margin of Safety (%)'),
            'threshold_summary': self.analysis.get('Threshold Summary'),
            'sector': self.analysis.get('Sector'),
            'market_cap': self.analysis.get('Market Cap')
        }


# Module initialization
logger.info("Stock Analyzer module initialized")
logger.debug(f"Available thresholds: {list(THRESHOLDS.keys())}")
