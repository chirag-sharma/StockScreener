# Stock Screener v2.0 - Professional Python Package

A comprehensive stock screening and analysis tool with AI-powered insights for Indian equity markets.

## 🏗️ Architecture

This project follows professional Python packaging standards:

```
stock_screener/             # Main package
├── core/                   # Core screening logic
├── services/               # Business services
├── utils/                  # Utilities and helpers
├── cli/                    # Command-line interface
├── config/                 # Configuration files
└── dashboard/              # Dashboard functionality

scripts/                    # Entry point scripts
data/                       # Input/output data
tests/                      # Test suite
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd StockScreener

# Install dependencies
pip install -r requirements.txt

# Configure API keys (optional for AI features)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Usage

#### Unified Screening (Default)
Comprehensive screening with AI analysis:
```bash
python scripts/run_screener.py
```

#### Legacy Mode
Traditional detailed analysis:
```bash
python scripts/run_screener.py --legacy
```

#### Custom Configuration
```bash
python scripts/run_screener.py --config stock_screener/config/nifty50.properties
```

#### Version Information
```bash
python scripts/run_screener.py --version
```

### 🎯 Price Prediction Tool

#### Quick Price Prediction
```bash
python scripts/predict_prices.py RELIANCE.NS
```

#### Comprehensive Analysis (All Methods)
```bash
python scripts/predict_prices.py --comprehensive RELIANCE.NS
```

#### Batch Predictions
```bash
python scripts/predict_prices.py --batch RELIANCE.NS TCS.NS INFY.NS
```

## 📊 Features

### Core Screening Engine
- **Multi-scope Analysis**: NIFTY 50, 100, 500, sectoral indices
- **Value Metrics**: PE, PB, Debt/Equity, ROE, Current Ratio
- **Quality Indicators**: Promoter holding, cash flow, margins
- **Growth Metrics**: EPS growth, revenue growth, ROA

### AI-Powered Analysis
- **Investment Thesis**: Automated analysis using Graham & Buffett principles
- **Risk Assessment**: Comprehensive risk evaluation
- **News Integration**: Real-time news sentiment analysis
- **Target Pricing**: AI-driven valuation estimates

### 🎯 Advanced Price Prediction (NEW!)
- **Multi-Method Forecasting**: Technical, fundamental, ML, time series analysis
- **30-Day Price Targets**: Forward-looking price predictions with confidence scores
- **Ensemble Predictions**: Combines multiple methods for enhanced accuracy
- **Risk Assessment**: Volatility analysis and drawdown calculations
- **Standalone Tool**: Dedicated price prediction CLI for individual stocks

### Professional Output
- **Excel Reports**: Formatted analysis with conditional highlighting
- **Investment Grades**: Strong Buy/Buy/Hold recommendations
- **Readiness Scores**: Investment readiness out of 100
- **Visual Dashboard**: Interactive analysis dashboard

## 🛠️ Configuration

### Screener Configuration
Edit `stock_screener/config/screener_config.properties`:

```properties
# Screening scope
scope=nifty_test

# AI Configuration
ai.enabled=true
ai.provider=openai
ai.include_news=true

# Output settings
output.excel.enabled=true
output.dashboard.enabled=false
```

### Custom Stock Lists
Add ticker lists in `data/input/tickers/`:
- `nifty_50.json` - NIFTY 50 stocks
- `nifty_100.json` - NIFTY 100 stocks  
- `custom_list.json` - Your custom stocks

## � Analysis Workflow

1. **Data Collection**: Fetch financial metrics from screener.in
2. **Basic Screening**: Apply value investing filters
3. **AI Analysis**: Comprehensive evaluation with news integration
4. **Report Generation**: Professional Excel output with formatting
5. **Dashboard Display**: Interactive visualization (optional)

## 🧪 Testing

Run the test suite:
```bash
python tests/run_tests.py
```

## 🔧 Development

### Package Structure
- `stock_screener.core.screener`: Main screening engine
- `stock_screener.services`: Business logic services
- `stock_screener.utils`: Helper utilities
- `stock_screener.cli`: Command-line interface

### Adding New Features
1. Implement in appropriate package module
2. Add configuration options
3. Update CLI arguments if needed
4. Add tests in `tests/` directory

## 📄 Output Files

- `comprehensive_analysis.xlsx` - Full unified analysis **with price predictions**
- `detailed_analysis.xlsx` - Legacy detailed analysis  
- `value_analysis.xlsx` - Basic value screening results

### New Price Prediction Columns
- **Predicted Price (30d)**: 30-day forward price target
- **Price Change %**: Expected percentage change
- **Prediction Confidence**: Model confidence (0-100%)
- **Prediction Method**: Method used for prediction
- **Target Price**: AI-enhanced target combining all factors

## 🤖 AI Providers

Supported AI providers:
- **OpenAI**: GPT models for analysis
- **Anthropic**: Claude models for analysis
- **Auto**: Automatic provider detection

## 📋 Requirements

- Python 3.8+
- pandas, openpyxl, requests
- Optional: openai, anthropic (for AI features)

## 🏷️ Version

Current version: 2.0.0

## 📞 Support

For issues or questions, please check the documentation or create an issue in the repository.
- **Target Price**: AI-estimated fair value
- **Growth Catalysts**: Upcoming opportunities
- **Key Risks**: Specific concerns from actual data

### Real Data Sources
- Financial metrics from market APIs
- Recent news from Economic Times, MoneyControl, News API, Google News
- Company fundamentals analysis

## 🔧 Configuration Options

### AI Providers
```properties
ai_provider = openai     # Most reliable, requires API key
ai_provider = gemini     # Free tier available
ai_provider = claude     # High quality analysis  
ai_provider = ollama     # Local/offline analysis
ai_provider = auto       # Auto-detect best available
```

### Investment Criteria
```properties
roe_min = 15
pe_ratio_max = 20
debt_to_equity_max = 1
current_ratio_min = 1.5
# ... and 20+ more criteria
```

## 🗂️ File Structure

```
StockScreener/
├── unified_screener.py          # ⭐ Main unified workflow
├── run_screener.sh              # Simple launcher
├── config/
│   └── screener_config.properties  # All settings here
├── data/output/
│   └── comprehensive_analysis.xlsx # Final report
├── detailed_analysis.py         # Legacy detailed analysis (still works)
└── src/screener.py              # Legacy basic screener (still works)
```

## 🔑 API Keys Setup

Create `.env` file:
```bash
# OpenAI (recommended)
OPENAI_API_KEY=sk-proj-your-key-here

# Google Gemini (free tier)
GOOGLE_API_KEY=your-google-key

# Optional: News API for enhanced news
NEWS_API_KEY=your-news-api-key
```

## 🆚 Old vs New Workflow

### ❌ Old Way (2 steps, 2 files)
```bash
python src/screener.py          # Produces value_analysis.xlsx
python detailed_analysis.py     # Produces detailed_analysis.xlsx
```

### ✅ New Way (1 step, 1 file)  
```bash
python unified_screener.py      # Produces comprehensive_analysis.xlsx
```

## 🎯 Use Cases

### Quick Screening
```bash
# Edit config/screener_config.properties
ai_analysis_enabled = false
python unified_screener.py
```

### Full AI Analysis  
```bash
# Edit config/screener_config.properties
ai_analysis_enabled = true
ai_provider = openai
news_integration_enabled = true
python unified_screener.py
```

### Automated/Production Use
```bash
# No command line arguments needed - everything in config
./run_screener.sh
```

## 📈 Sample Output

```
🎯 COMPREHENSIVE SCREENING COMPLETE!
📊 Total stocks analyzed: 50
📈 Average readiness score: 65.2/100
🧠 Average AI value score: 6.3/10
⭐ Excellent candidates: 8

📋 Investment Recommendations:
   - Strong Buy: 3
   - Buy: 5
   - Hold/Others: 42

📄 Comprehensive report: data/output/comprehensive_analysis.xlsx
```

## 🛠️ Legacy Support

Old scripts still work for backward compatibility:
- `python src/screener.py` - Basic screening only
- `python detailed_analysis.py --openai` - Command-line AI analysis

## 📝 Value Investing Principles

Built-in benchmarks following Graham & Buffett:
- **PE Ratio**: Excellent (8-15), Good (15-20)
- **Price to Book**: Excellent (0.5-1.5), Good (1.5-2.5)  
- **ROE**: Excellent (20-50%), Good (15-20%)
- **Current Ratio**: Excellent (2.0-3.0), Good (1.5-2.0)
- **Debt/Equity**: Excellent (0-0.3), Good (0.3-0.6)

---

**🎉 Result**: One command, one configuration file, one comprehensive report with real AI analysis!
