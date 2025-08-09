"""
This module provides the StockAnalyzer class for fetching and analyzing stock data using yfinance.
It applies financial thresholds and returns a structured analysis result.
"""
import yfinance as yf
from stock_screener.core.constants import THRESHOLDS
import logging

class StockAnalyzer:
    """
    Analyzes a stock's financial data and evaluates it against predefined thresholds.
    """
    def __init__(self, symbol):
        """
        Initialize the analyzer with a stock symbol.
        Args:
            symbol (str): The stock ticker symbol.
        """
        self.symbol = symbol
        self.info = {}
        self.analysis = {}

    def fetch_data(self):
        """
        Fetch stock data using yfinance and store it in self.info.
        Logs an error if data cannot be fetched.
        """
        try:
            stock = yf.Ticker(self.symbol)
            self.info = stock.info
        except Exception as e:
            logging.error(f"Failed to fetch data for {self.symbol}: {e}")
            self.info = {}

    def analyze(self):
        """
        Analyze the fetched stock data and return a dictionary of metrics.
        Returns None if no data is available.
        """
        if not self.info:
            return None

        def safe_get(key, transform=None):
            val = self.info.get(key)
            if val is not None and transform:
                try:
                    return transform(val)
                except Exception as e:
                    logging.error(f"Error transforming {key}: {e}")
                    return None
            return val

        def calculate_price_to_cash_flow():
            """Calculate Price to Cash Flow ratio using Market Cap and Operating Cash Flow"""
            market_cap = safe_get('marketCap')
            revenue = safe_get('totalRevenue')
            op_margin = safe_get('operatingMargins')

            if all(v is not None for v in [market_cap, revenue, op_margin]):
                operating_cash_flow = revenue * op_margin
                if operating_cash_flow > 0:  # Avoid division by zero or negative values
                    return market_cap / operating_cash_flow
            return None

        def calculate_net_profit_margin():
            """Calculate Net Profit Margin with fallback options"""
            # First try using profitMargins directly
            profit_margins = safe_get('profitMargins')
            if profit_margins is not None:
                return profit_margins * 100

            # Fallback to calculation from raw numbers
            net_income = safe_get('netIncomeToCommon')
            revenue = safe_get('totalRevenue')
            if net_income is not None and revenue is not None and revenue != 0:
                return (net_income / revenue) * 100
            return None

        # Collect key financial metrics
        self.analysis = {
            'Symbol': self.symbol,
            'PE Ratio': safe_get('trailingPE'),
            'Debt/Equity': safe_get('debtToEquity'),
            'ROE': safe_get('returnOnEquity', lambda x: x * 100),
            'Current Ratio': safe_get('currentRatio'),
            'Price to Book': safe_get('priceToBook'),
            'Promoter Holding': safe_get('heldPercentInsiders', lambda x: x * 100),
            'Price to Cash Flow': calculate_price_to_cash_flow(),
            'Quick Ratio': safe_get('quickRatio'),
            'Interest Coverage Ratio': safe_get('ebitda', lambda x: x / self.info.get('totalInterestExpense', 1)),
            'Free Cash Flow': safe_get('freeCashflow'),
            'EPS Growth (%)': safe_get('earningsQuarterlyGrowth', lambda x: x * 100),
            'Return on Assets (ROA)': safe_get('returnOnAssets', lambda x: x * 100),
            'Net Profit Margin': calculate_net_profit_margin(),
            'Operating Margin': safe_get('operatingMargins', lambda x: x * 100),
            'Cash Conversion Ratio': safe_get('freeCashflow', lambda x: x / self.info.get('netIncome', 1)),
            'Pledged Shares (%)': None,  # Not available in yfinance
            'EV/EBITDA': safe_get('enterpriseToEbitda'),
            'Revenue Growth (%)': safe_get('revenueGrowth', lambda x: x * 100)
        }

        # Log calculations for debugging
        logging.info(f"\nCalculated metrics for {self.symbol}:")
        if self.analysis['Price to Cash Flow'] is not None:
            logging.info(f"Price to Cash Flow: {self.analysis['Price to Cash Flow']:.2f}")
        if self.analysis['Net Profit Margin'] is not None:
            logging.info(f"Net Profit Margin: {self.analysis['Net Profit Margin']:.2f}%")

        # Apply thresholds from constants
        self.analysis['PE Ratio Pass'] = self._check('PE Ratio', '<', THRESHOLDS['pe_ratio_max'])
        self.analysis['Debt/Equity Pass'] = self._check('Debt/Equity', '<', THRESHOLDS['debt_to_equity_max'])
        self.analysis['ROE Pass'] = self._check('ROE', '>', THRESHOLDS['roe_min'])
        self.analysis['Current Ratio Pass'] = self._check('Current Ratio', '>', THRESHOLDS['current_ratio_min'])
        self.analysis['Price to Book Pass'] = self._check('Price to Book', '<', THRESHOLDS['price_to_book_max'])
        self.analysis['Promoter Holding Pass'] = self._check('Promoter Holding', '>', THRESHOLDS['promoter_holding_min'])
        self.analysis['Price to Cash Flow Pass'] = self._check('Price to Cash Flow', '<', THRESHOLDS['price_to_cash_flow_max'])
        self.analysis['Quick Ratio Pass'] = self._check('Quick Ratio', '>', THRESHOLDS['quick_ratio_min'])
        self.analysis['Interest Coverage Ratio Pass'] = self._check('Interest Coverage Ratio', '>', THRESHOLDS['interest_coverage_min'])
        self.analysis['Free Cash Flow Pass'] = self._check('Free Cash Flow', '>', THRESHOLDS['free_cash_flow_min'])
        self.analysis['EPS Growth (%) Pass'] = self._check('EPS Growth (%)', '>', THRESHOLDS['eps_growth_min'])
        self.analysis['Return on Assets (ROA) Pass'] = self._check('Return on Assets (ROA)', '>', THRESHOLDS['roa_min'])
        self.analysis['Net Profit Margin Pass'] = self._check('Net Profit Margin', '>', THRESHOLDS['net_profit_margin_min'])
        self.analysis['Operating Margin Pass'] = self._check('Operating Margin', '>', THRESHOLDS['operating_margin_min'])
        self.analysis['Cash Conversion Ratio Pass'] = self._check('Cash Conversion Ratio', '>', THRESHOLDS['cash_conversion_min'])
        self.analysis['Pledged Shares Pass'] = self._check('Pledged Shares (%)', '<', THRESHOLDS['pledged_shares_max'])
        self.analysis['EV/EBITDA Pass'] = self._check('EV/EBITDA', '<', THRESHOLDS['ev_ebitda_max'])
        self.analysis['Revenue Growth (%) Pass'] = self._check('Revenue Growth (%)', '>', THRESHOLDS['revenue_growth_min'])

        # Calculate margin of safety
        free_cash_flow = self.info.get('freeCashflow')
        market_value = self.info.get('marketCap')

        if free_cash_flow and market_value and market_value != 0:
            intrinsic_value = free_cash_flow * 10  # Simple 10x FCF estimate
            mos = (intrinsic_value - market_value) / market_value * 100
            self.analysis['Margin of Safety (%)'] = round(mos, 2)
        else:
            self.analysis['Margin of Safety (%)'] = 'N/A'

        # Calculate weighted value investing score
        self.analysis['Value Score'] = self._calculate_weighted_score()
        
        # Overall investment suggestion based on weighted score
        score = self.analysis['Value Score']
        if score >= 90:
            self.analysis['Investment Recommendation'] = 'Strong Buy'
        elif score >= 70:
            self.analysis['Investment Recommendation'] = 'Buy'
        elif score >= 50:
            self.analysis['Investment Recommendation'] = 'Hold'
        elif score >= 30:
            self.analysis['Investment Recommendation'] = 'Weak Hold'
        else:
            self.analysis['Investment Recommendation'] = 'Avoid'

        return self.analysis

    def _calculate_weighted_score(self):
        """
        Calculate weighted value investing score based on fundamental principles.
        Returns a score between 0-100 based on weighted criteria.
        """
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
            return round(final_score, 1)
        else:
            return 0.0
    
    def _score_metric(self, metric, value):
        """
        Score individual metrics on a 0-100 scale based on value investing criteria.
        Higher scores indicate better value/quality.
        """
        # Handle string values
        if isinstance(value, str):
            return 0.0
            
        try:
            value = float(value)
        except (ValueError, TypeError):
            return 0.0
        
        # Scoring logic for each metric
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
        return 50.0

    def _check(self, key, condition, threshold):
        val = self.analysis.get(key)
        if val is None:
            return False
        if condition == '<':
            return val < threshold
        if condition == '>':
            return val > threshold
        return False
