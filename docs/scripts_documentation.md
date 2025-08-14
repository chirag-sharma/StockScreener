# Stock Screener - Simplified Launchers

This directory contains simplified scripts to run different types of stock analysis.

## 🚀 Quick Start Scripts

### 1. Main Launcher (Recommended)
```bash
python scripts/launcher.py
```
**Interactive menu with options for:**
- AI-powered analysis (real OpenAI)
- Basic analysis (fast, rule-based)  
- Dashboard (web interface)

### 2. AI-Powered Analysis
```bash
python scripts/run_ai_screener.py
```
**Features:**
- ✅ Real OpenAI GPT integration
- ✅ News sentiment analysis
- ✅ Price predictions (6-12 months)
- ✅ Comprehensive Excel reports
- ⏱️ Takes 1-2 minutes

### 3. Basic Analysis (Fast)
```bash
python scripts/run_basic_screener.py
```
**Features:**
- ✅ Rule-based recommendations
- ✅ Financial metrics analysis
- ✅ Excel report generation
- ⚡ Completes in 3-5 seconds

## 📊 Analysis Comparison

| Feature | Basic Analysis | AI Analysis |
|---------|---------------|-------------|
| **Speed** | ⚡ 3-5 seconds | 🕐 1-2 minutes |
| **AI Integration** | ❌ Rule-based | ✅ Real OpenAI |
| **News Analysis** | ❌ None | ✅ Real-time |
| **Price Predictions** | ❌ Simple | ✅ AI-powered |
| **Investment Thesis** | ❌ None | ✅ Comprehensive |
| **Report Quality** | 📊 Basic | 🧠 Professional |

## 🔧 Requirements

1. **Virtual Environment**: `.venv` with all dependencies
2. **API Keys**: OpenAI API key in `.env` file (for AI analysis)
3. **Configuration**: `config/screener_config.properties` properly set

## 📁 Output Files

All analysis results are saved in:
- **Location**: `data/output/`
- **Format**: Excel files with multiple sheets
- **Contains**: Financial metrics, recommendations, AI analysis

## 🎯 Usage Examples

**Quick Analysis:**
```bash
python scripts/run_basic_screener.py
```

**Comprehensive AI Analysis:**
```bash
python scripts/run_ai_screener.py
```

**Interactive Experience:**
```bash
python scripts/launcher.py
# Select option 1 for AI analysis
# Select option 2 for basic analysis  
# Select option 3 for dashboard
```

## ✅ Success Indicators

**Basic Analysis Success:**
```
✅ Basic analysis completed!
📄 Check the data/output/ directory for Excel reports
```

**AI Analysis Success:**
```
🎯 COMPREHENSIVE SCREENING COMPLETE!
🤖 Enhanced with comprehensive AI analysis
📄 Comprehensive report: data/output/comprehensive_analysis.xlsx
```

## 🆘 Troubleshooting

**Virtual Environment Issues:**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

**API Key Issues:**
- Ensure `.env` file has `OPENAI_API_KEY=your_key_here`
- Check API key has sufficient credits

**Configuration Issues:**
- Verify `config/screener_config.properties` exists
- Check sector setting matches available ticker files

## 🎉 What's Different?

These simplified scripts replace the need to:
- Remember complex module paths
- Use subprocess calls manually
- Handle virtual environment activation
- Deal with configuration issues

Just run the script and get results! 🚀
