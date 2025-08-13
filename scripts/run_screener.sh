#!/bin/bash

# Stock Screener Launcher
# ======================
# Professional entry point for the unified stock screener

echo "🚀 Starting Professional Stock Screener with AI Analysis..."
echo "📊 Configuration: stock_screener/config/screener_config.properties"
echo ""

# Run the unified screener using the professional entry point
python scripts/run_screener.py "$@"

echo ""
echo "✅ Analysis complete! Check the generated Excel file for results."
