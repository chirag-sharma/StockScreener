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
            return None

        def safe_get(key, transform=None):
            val = self.info.get(key)
            if val is not None and transform:
                return transform(val)
            return val

        self.analysis = {
            'Symbol': self.symbol,
            'PE Ratio': safe_get('trailingPE'),
            'Debt/Equity': safe_get('debtToEquity'),
            'ROE': safe_get('returnOnEquity', lambda x: x * 100),
            'Current Ratio': safe_get('currentRatio'),
            'Price to Book': safe_get('priceToBook'),
            'Promoter Holding': safe_get('heldPercentInsiders', lambda x: x * 100)
        }

        # Apply thresholds from constants
        self.analysis['PE Ratio Pass'] = self._check('PE Ratio', '<', THRESHOLDS['pe_ratio_max'])
        self.analysis['Debt/Equity Pass'] = self._check('Debt/Equity', '<', THRESHOLDS['debt_to_equity_max'])
        self.analysis['ROE Pass'] = self._check('ROE', '>', THRESHOLDS['roe_min'])
        self.analysis['Current Ratio Pass'] = self._check('Current Ratio', '>', THRESHOLDS['current_ratio_min'])
        self.analysis['Price to Book Pass'] = self._check('Price to Book', '<', THRESHOLDS['price_to_book_max'])
        self.analysis['Promoter Holding Pass'] = self._check('Promoter Holding', '>', THRESHOLDS['promoter_holding_min'])

        # Optional: Overall investment suggestion
        pass_count = sum([self.analysis[k] for k in self.analysis if k.endswith('Pass')])
        if pass_count >= 5:
            self.analysis['Investment Recommendation'] = 'Strong Buy'
        elif 3 <= pass_count < 5:
            self.analysis['Investment Recommendation'] = 'Hold'
        else:
            self.analysis['Investment Recommendation'] = 'Avoid'

        return self.analysis

    def _check(self, key, condition, threshold):
        val = self.analysis.get(key)
        if val is None:
            return False
        if condition == '<':
            return val < threshold
        if condition == '>':
            return val > threshold
        return False
