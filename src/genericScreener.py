import yfinance as yf
import pandas as pd
from configparser import ConfigParser

def load_symbols_from_properties():
    config = ConfigParser()
    config.read('../config/screener_config.properties')
    ticker_file_path = config.get('DEFAULT', 'tickerFile', fallback='../config/nifty50.properties')

    ticker_config = ConfigParser()
    ticker_config.read(ticker_file_path)
    symbols = dict(ticker_config['TICKERS'])
    return symbols

def analyze_stock(symbol, display_name):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        dy = info.get('dividendYield', 0.0)
        roe = info.get('returnOnEquity')
        de = info.get('debtToEquity')
        market_cap = info.get('marketCap')

        result = {'Stock': display_name, 'Symbol': symbol, 'P/E Ratio': pe, 'P/B Ratio': pb,
                  'Dividend Yield (%)': round(dy * 100, 2) if dy else None,
                  'ROE (%)': round(roe * 100, 2) if roe else None, 'Debt to Equity': de, 'Market Cap': market_cap,
                  'Pass Value Criteria': (
                          (pe is not None and pe < 20) and
                          (pb is not None and pb < 3) and
                          (dy is not None and dy > 0.01) and
                          (roe is not None and roe > 10) and
                          (de is not None and de < 1)
                  )}

        # Value Investing Pass/Fail

        return result
    except Exception as e:
        return {
            'Stock': display_name,
            'Symbol': symbol,
            'Error': str(e)
        }

def main():
    symbols = load_symbols_from_properties()
    results = []

    for display_name, symbol in symbols.items():
        print(f"Processing {display_name}...")
        result = analyze_stock(symbol, display_name)
        results.append(result)

    df = pd.DataFrame(results)

    # Save to Excel
    df.to_excel('../data/output/value_investment_report.xlsx', index=False)
    print("\n✅ Excel report generated at '../data/output/value_investment_report.xlsx'")

if __name__ == '__main__':
    main()
