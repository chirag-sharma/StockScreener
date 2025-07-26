from services.excelExporter import ExcelExporter
from services.stockAnalyzer import StockAnalyzer
from utils.configLoader import load_symbols_from_config
from utils.tickerLoader import load_tickers

def main():
    # Load list of stock tickers (from JSON or config)
    symbols = load_symbols_from_config()
    #ticker_list = load_tickers("../data/input/tickers.json")

    all_results = []

    for symbol in symbols:
        analyzer = StockAnalyzer(symbol)
        analyzer.fetch_data()
        result = analyzer.analyze()
        if result:
            all_results.append(result)

    if all_results:
        exporter = ExcelExporter("../data/output/value_analysis.xlsx")
        exporter.write_data(all_results)
        print("✅ Analysis complete. Excel file created.")
    else:
        print("⚠️ No valid data to write.")

if __name__ == "__main__":
    main()
