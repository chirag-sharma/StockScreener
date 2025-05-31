import yfinance as yf
import pandas as pd

# List of Nifty 50 stock symbols on Yahoo Finance (India)
nifty_50_symbols = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS",
    "AXISBANK.NS", "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "MARUTI.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "HINDALCO.NS", "TITAN.NS", "COALINDIA.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS", "NTPC.NS", "CIPLA.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "GRASIM.NS", "BRITANNIA.NS", "M&M.NS", "DIVISLAB.NS", "HDFCLIFE.NS", "TATACONSUM.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "SBILIFE.NS", "SHREECEM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "UPL.NS", "BAJAJ-AUTO.NS"
]

# Empty list to collect screened data
screened_stocks = []

for symbol in nifty_50_symbols:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # Extract required financial metrics
        roe = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0
        pe = info.get("trailingPE", 0)
        debt_equity = info.get("debtToEquity", 0)
        profit_margin = info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0
        revenue_growth = info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0
        market_cap_cr = info.get("marketCap", 0) / 1e7  # Convert to Crores

        # Apply screening filters
        if (
            roe > 15 and
            pe < 25 and
            debt_equity < 100 and  # since debtToEquity is in %
            profit_margin > 10 and
            revenue_growth > 10 and
            market_cap_cr > 10000
        ):
            screened_stocks.append({
                "Symbol": symbol,
                "ROE (%)": round(roe, 2),
                "PE Ratio": round(pe, 2),
                "Debt to Equity (%)": round(debt_equity, 2),
                "Net Profit Margin (%)": round(profit_margin, 2),
                "Revenue Growth YoY (%)": round(revenue_growth, 2),
                "Market Cap (Cr)": round(market_cap_cr, 2)
            })

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# Convert to DataFrame
df = pd.DataFrame(screened_stocks)

# Save to Excel
output_file = "../data/nifty50_screener_results.xlsx"
df.to_excel(output_file, index=False)

print(f"\n✅ Screener completed. {len(df)} stocks matched the criteria.")
print(f"📁 Results saved to: {output_file}")
