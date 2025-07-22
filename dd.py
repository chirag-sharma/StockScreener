import pandas as pd
import json
from datetime import datetime

# Load the CSV file
csv_file_path = "/data/ind_nifty500list.csv"
df = pd.read_csv(csv_file_path)

# Extract symbols and append '.NS'
symbols = df['Symbol'].dropna().astype(str).str.strip().str.upper()
symbols_with_suffix = [symbol + ".NS" for symbol in symbols]

# Create JSON structure
output_data = {
    "date": datetime.today().strftime("%Y-%m-%d"),
    "tickers": symbols_with_suffix
}

# Save to JSON file
json_file_path = "/data/nifty_500_tickers_generated.json"
with open(json_file_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ JSON saved to: {json_file_path}")
