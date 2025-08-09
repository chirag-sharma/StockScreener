#!/usr/bin/env python3
"""Entry point script for stock screener."""

import sys
from pathlib import Path

# Add package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_screener.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
