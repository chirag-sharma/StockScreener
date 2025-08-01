import yfinance as yf
from src.constants import THRESHOLDS

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = {}
        self.analysis = {}

    def fetch_data(self):
        try:
            stock = yf.Ticker(self.symbol)
            self.info = stock.info
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {self.symbol}: {e}")
            self.info = {}

    def analyze(self):
        if not self.info:
            print(f"[WARN] No data fetched for {self.symbol}")
            return {'Symbol': self.symbol, 'Investment Recommendation': 'Insufficient data'}

        def safe_get(key, transform=None):
            try:
                val = self.info.get(key)
                if val is None:
                    return None
                return transform(val) if transform else val
            except Exception as e:
                print(f"[WARN] Failed to process key '{key}' for {self.symbol}: {e}")
                return None

        self.analysis = {
            'Symbol': self.symbol,
            'PE Ratio': safe_get('trailingPE'),
            'Debt/Equity': safe_get('debtToEquity'),
            'ROE': safe_get('returnOnEquity', lambda x: x * 100),
            'Current Ratio': safe_get('currentRatio'),
            'Price to Book': safe_get('priceToBook'),
            'Promoter Holding': safe_get('heldPercentInsiders', lambda x: x * 100),
            'EPS Growth (%)': safe_get('earningsQuarterlyGrowth', lambda x: x * 100),
            'Return on Assets (ROA)': safe_get('returnOnAssets', lambda x: x * 100),
            'Dividend Yield': safe_get('dividendYield', lambda x: x * 100),
            'Free Cash Flow': safe_get('freeCashflow'),
            #'Price to Cash Flow': safe_get('priceToCashflow'),
            'Enterprise Value': safe_get('enterpriseValue'),
            'EV/EBITDA': safe_get('enterpriseToEbitda'),
            #'Interest Coverage Ratio': safe_get('ebitda', lambda x: x / self.info.get('totalInterestExpense', 1) if self.info.get('totalInterestExpense') else None),
            'Quick Ratio': safe_get('quickRatio'),
            #'Net Profit Margin': safe_get('netMargins', lambda x: x * 100),
            'Operating Margin': safe_get('operatingMargins', lambda x: x * 100),
            #'Cash Conversion Ratio': safe_get('freeCashflow', lambda x: x / self.info.get('netIncome', 1) if self.info.get('netIncome') else None),
            'Dividend Payout Ratio': safe_get('payoutRatio', lambda x: x * 100),
            #'Market Cap': safe_get('marketCap', lambda x: x / 1e7)  # in Crores
        }

        # Check if all (or nearly all) key metrics are missing
        numeric_keys = [k for k in self.analysis.keys() if k != 'Symbol']
        missing_count = sum(1 for k in numeric_keys if self.analysis[k] is None)
        if missing_count >= len(numeric_keys) - 1:
            self.analysis['Investment Recommendation'] = 'Insufficient data'
            return self.analysis

        # Threshold checks
        for key, threshold_key in {
            'PE Ratio': 'pe_ratio_max',
            'Debt/Equity': 'debt_to_equity_max',
            'ROE': 'roe_min',
            'Current Ratio': 'current_ratio_min',
            'Price to Book': 'price_to_book_max',
            'Promoter Holding': 'promoter_holding_min',
            'EPS Growth (%)': 'eps_growth_min',
            'Return on Assets (ROA)': 'roa_min',
            'Dividend Yield': 'dividend_yield_min',
            'Free Cash Flow': 'free_cash_flow_min',
            #'Price to Cash Flow': 'price_to_cash_flow_max',
            'EV/EBITDA': 'ev_ebitda_max',
            #'Interest Coverage Ratio': 'interest_coverage_min',
            'Quick Ratio': 'quick_ratio_min',
            #'Net Profit Margin': 'net_profit_margin_min',
            'Operating Margin': 'operating_margin_min',
            #'Cash Conversion Ratio': 'cash_conversion_ratio_min',
            'Dividend Payout Ratio': 'dividend_payout_ratio_max',
            #'Market Cap': 'market_cap_min'
        }.items():
            threshold = THRESHOLDS.get(threshold_key)
            if threshold is None:
                print(f"[WARN] Threshold for '{key}' not defined: missing key '{threshold_key}'")
                self.analysis[f'{key} Pass'] = None  # or False, based on preference
                continue

            if 'max' in threshold_key:
                self.analysis[f'{key} Pass'] = self._check(key, '<', threshold)
            else:
                self.analysis[f'{key} Pass'] = self._check(key, '>', threshold)

        # Scoring
        pass_count = sum([self.analysis[k] for k in self.analysis if k.endswith('Pass') and isinstance(self.analysis[k], bool)])
        if pass_count >= 12:
            self.analysis['Investment Recommendation'] = 'Strong Buy'
        elif 7 <= pass_count < 12:
            self.analysis['Investment Recommendation'] = 'Hold'
        else:
            self.analysis['Investment Recommendation'] = 'Avoid'

        return self.analysis

    def _check(self, key, condition, threshold):
        try:
            val = self.analysis.get(key)
            if val is None:
                return False
            if condition == '<':
                return val < threshold
            if condition == '>':
                return val > threshold
        except Exception as e:
            print(f"[ERROR] While checking {key}: {e}")
            return False
