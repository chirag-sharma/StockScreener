# src/utils.py
def load_tickers_from_properties(filepath: str) -> list:
    tickers = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("tickers"):
                _, tickers_str = line.strip().split('=')
                tickers = [t.strip() for t in tickers_str.split(',')]
    return tickers
