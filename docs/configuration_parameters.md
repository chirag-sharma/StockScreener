# Configuration Parameters Reference

## Complete List of Available Parameters in screener_config.properties

### Section: [DEFAULT]

#### Sector Configuration
- `sector` - Target sector for analysis (e.g., nifty_50, nifty_100, nifty_test)

#### AI Analysis Settings
- `ai_analysis_enabled` - Enable/disable AI analysis (true/false)
- `ai_provider` - AI provider selection (openai, gemini, claude, auto)
- `news_integration_enabled` - Enable news sentiment analysis (true/false)
- `news_days_back` - Days to look back for news (default: 30)

#### Financial Thresholds (21 parameters)

##### Profitability Thresholds
- `roe_min` - Minimum Return on Equity (%) [Default: 15.0]
- `roa_min` - Minimum Return on Assets (%) [Default: 10.0]
- `net_profit_margin_min` - Minimum Net Profit Margin (%) [Default: 10.0]
- `operating_margin_min` - Minimum Operating Margin (%) [Default: 15.0]

##### Valuation Thresholds
- `pe_ratio_max` - Maximum PE Ratio [Default: 20.0]
- `price_to_book_max` - Maximum Price to Book Ratio [Default: 3.0]
- `price_to_cash_flow_max` - Maximum Price to Cash Flow Ratio [Default: 15.0]
- `ev_ebitda_max` - Maximum EV/EBITDA Ratio [Default: 10.0]

##### Financial Health Thresholds
- `de_ratio_max` - Maximum Debt to Equity Ratio [Default: 1.0]
- `current_ratio_min` - Minimum Current Ratio [Default: 1.5]
- `quick_ratio_min` - Minimum Quick Ratio [Default: 1.0]
- `interest_coverage_min` - Minimum Interest Coverage Ratio [Default: 2.0]

##### Growth Thresholds
- `eps_growth_min` - Minimum EPS Growth (%) [Default: 10.0]
- `revenue_growth_min` - Minimum Revenue Growth (%) [Default: 10.0]

##### Management & Ownership
- `promoter_holding_min` - Minimum Promoter Holding (%) [Default: 50.0]
- `pledged_shares_max` - Maximum Pledged Shares (%) [Default: 10.0]

##### Size & Quality Filters
- `market_cap_min` - Minimum Market Cap (Crores) [Default: 1000.0]
- `free_cash_flow_min` - Minimum Free Cash Flow [Default: 0.0]
- `cash_conversion_min` - Minimum Cash Conversion Ratio [Default: 1.0]

##### Dividend Policy
- `dividend_yield_min` - Minimum Dividend Yield (%) [Default: 2.0]
- `dividend_payout_min` - Minimum Dividend Payout Ratio (%) [Default: 40.0]

## Parameter-to-Constant Mapping

| Config Parameter | Constants.py Key | Default Value |
|-----------------|------------------|---------------|
| `roe_min` | `ROE_MIN` | 15.0 |
| `pe_ratio_max` | `PE_RATIO_MAX` | 20.0 |
| `de_ratio_max` | `DE_RATIO_MAX` | 1.0 |
| `current_ratio_min` | `CURRENT_RATIO_MIN` | 1.5 |
| `price_to_book_max` | `PRICE_TO_BOOK_MAX` | 3.0 |
| `promoter_holding_min` | `PROMOTER_HOLDING_MIN` | 50.0 |
| `price_to_cash_flow_max` | `PRICE_TO_CASH_FLOW_MAX` | 15.0 |
| `quick_ratio_min` | `QUICK_RATIO_MIN` | 1.0 |
| `interest_coverage_min` | `INTEREST_COVERAGE_MIN` | 2.0 |
| `free_cash_flow_min` | `FREE_CASH_FLOW_MIN` | 0.0 |
| `eps_growth_min` | `EPS_GROWTH_MIN` | 10.0 |
| `roa_min` | `ROA_MIN` | 10.0 |
| `net_profit_margin_min` | `NET_PROFIT_MARGIN_MIN` | 10.0 |
| `operating_margin_min` | `OPERATING_MARGIN_MIN` | 15.0 |
| `cash_conversion_min` | `CASH_CONVERSION_MIN` | 1.0 |
| `ev_ebitda_max` | `EV_EBITDA_MAX` | 10.0 |
| `market_cap_min` | `MARKET_CAP_MIN` | 1000.0 |
| `pledged_shares_max` | `PLEDGED_SHARES_MAX` | 10.0 |
| `revenue_growth_min` | `REVENUE_GROWTH_MIN` | 10.0 |
| `dividend_yield_min` | `DIVIDEND_YIELD_MIN` | 2.0 |
| `dividend_payout_min` | `DIVIDEND_PAYOUT_MIN` | 40.0 |

## Status: ✅ All Parameters Consistent

- ✅ All 21 financial threshold parameters are present in both files
- ✅ Parameter names match exactly between config and constants
- ✅ Default values are consistent
- ✅ No typos or naming mismatches
- ✅ Successfully tested with StockAnalyzer

Last Updated: August 14, 2025
