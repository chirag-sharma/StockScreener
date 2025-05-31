import yfinance as yf
import pandas as pd
import datetime

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

screened_stocks = []

# Date range for technical indicators
end = datetime.datetime.now()
start = end - datetime.timedelta(days=300)

for symbol in nifty_50_symbols:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        roe = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0
        pe = info.get("trailingPE", 0)
        debt_equity = info.get("debtToEquity", 0)
        profit_margin = info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0
        revenue_growth = info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0
        market_cap_cr = info.get("marketCap", 0) / 1e7

        df = stock.history(start=start, end=end)
        if df.empty or len(df) < 200:
            continue

        close_price = df["Close"].iloc[-1]
        sma_50 = df["Close"].rolling(window=50).mean().iloc[-1]
        sma_200 = df["Close"].rolling(window=200).mean().iloc[-1]

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        # --- MACD Calculation ---
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]

        # Apply combined filters
        if (
            roe > 15 and
            pe < 25 and
            debt_equity < 100 and
            profit_margin > 10 and
            revenue_growth > 10 and
            market_cap_cr > 10000 and
            close_price > sma_50 and
            close_price > sma_200 and
            40 <= latest_rsi <= 70 and
            latest_macd > latest_signal  # Bullish MACD crossover
        ):
            screened_stocks.append({
                "Symbol": symbol,
                "ROE (%)": round(roe, 2),
                "PE Ratio": round(pe, 2),
                "Debt to Equity (%)": round(debt_equity, 2),
                "Net Profit Margin (%)": round(profit_margin, 2),
                "Revenue Growth YoY (%)": round(revenue_growth, 2),
                "Market Cap (Cr)": round(market_cap_cr, 2),
                "Price": round(close_price, 2),
                "50 DMA": round(sma_50, 2),
                "200 DMA": round(sma_200, 2),
                "RSI (14)": round(latest_rsi, 2),
                "MACD": round(latest_macd, 2),
                "Signal": round(latest_signal, 2)
            })

    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")

df = pd.DataFrame(screened_stocks)
df.to_excel("nifty50_screener_results.xlsx", index=False)

print(f"\n✅ Screener completed. {len(df)} stocks passed all filters.")
print("📁 Results saved to: nifty50_screener_results.xlsx")
