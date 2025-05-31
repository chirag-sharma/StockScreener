Sure! Here's the full content of the `README.md` file you can copy-paste directly into your project:

---

```markdown
# 📊 Stock Screener AI

This project is a Value Investing Stock Screener that uses financial data of Indian stocks (e.g., Nifty 50/100/500) to evaluate companies based on fundamental investment metrics. It highlights underperforming stocks and their failed investment criteria in a downloadable Excel file.

---

## 🧠 Features

- 📥 Loads ticker symbols from `.properties` files.
- 📈 Fetches stock fundamentals using Yahoo Finance or Screener.in.
- ✅ Applies value investing rules (checklist-style):
  - PE Ratio < 20
  - ROE > 15%
  - ROCE > 15%
  - Debt to Equity < 1
  - Current Ratio > 1.5
  - Consistent Profit Growth
  - Positive Free Cash Flow
- ❌ Highlights failed metrics in red per stock.
- 📤 Exports results to a styled Excel sheet for investor-friendly review.

---

## 🗂️ Project Structure

```

StockScreener/
│
├── config/
│   ├── screener\_config.properties  # points to the relevant ticker file
│   ├── nifty50.properties          # contains SYMBOL=SYMBOL.NS lines
│   ├── nifty500.properties         # generated from Screener.in
│
├── src/
│   ├── genericScreener.py          # main analysis script
│   ├── utils/
│       ├── value\_analysis.py       # contains analysis logic
│       ├── excel\_writer.py         # writes styled Excel output
│
├── output/
│   ├── value\_investing\_report.xlsx # generated results
│
├── README.md                       # you're here!

````

---

## 🔧 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/StockScreener.git
cd StockScreener
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install yfinance openpyxl requests beautifulsoup4 pandas
```

### 3. Configure Tickers

* Open `config/screener_config.properties` and set:

```properties
tickerFile=../config/nifty500.properties
```

* Each ticker file should follow this format:

```properties
[TICKERS]
RELIANCE=RELIANCE.NS
TCS=TCS.NS
...
```

---

## 🚀 Run the Screener

```bash
cd src
python genericScreener.py
```

> Output Excel will be saved as `output/value_investing_report.xlsx`

---

## 📘 Value Investing Criteria Used

| Metric         | Target Value |
| -------------- | ------------ |
| PE Ratio       | < 20         |
| ROE            | > 15%        |
| ROCE           | > 15%        |
| Debt to Equity | < 1          |
| Current Ratio  | > 1.5        |
| Profit Growth  | Positive     |
| Free Cash Flow | Positive     |

> ❌ Failed metrics are highlighted in **red** in the Excel output.

---

## 📡 Data Sources

* [Yahoo Finance](https://finance.yahoo.com/)
* [Screener.in](https://www.screener.in/screens/197055/nifty-500/)

---

## 🧪 Example

Generated Excel (partial preview):

| Symbol   | PE | ROE   | ROCE  | D/E | Curr. Ratio | FCF | ... |
| -------- | -- | ----- | ----- | --- | ----------- | --- | --- |
| RELIANCE | 25 | 10% ❌ | 12% ❌ | 0.5 | 1.8         | Yes |     |
| INFY     | 18 | 19% ✅ | 20% ✅ | 0.3 | 2.1         | Yes |     |

---

## 📌 Future Enhancements

* Add UI with Streamlit
* Enable custom filter configuration via YAML
* Support for sector-based screening
* Cloud-based deployment (AWS/Azure)

---

## 👨‍💻 Author

**Chirag \[or your name]**
*MBA Student, FinTech Enthusiast, Python Developer*

---

## 📄 License

This project is licensed under the MIT License.

```

---

Let me know if you'd like a version that includes images, shields, or badges for GitHub as well.
```
