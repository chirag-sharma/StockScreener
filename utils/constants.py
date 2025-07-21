# utils/constants.py

"""
This module contains all static constants used across the stock screener project.
Includes mapping of scope identifiers to Screener.in URLs for indices and sectors.
"""

# Scope to Screener URL map for index-based and sector-based stock groupings
SCOPE_URL_MAP = {
    # Index-based scopes
    "nifty_50": "https://www.screener.in/screens/507420/nifty-50-companies/",
    "nifty_100": "https://www.screener.in/screens/1263240/nifty-100/",
    "nifty_200": "https://www.screener.in/screens/1263241/nifty-200/",
    "nifty_500": "https://www.screener.in/screens/1263242/nifty-500/",
    "midcap_100": "https://www.screener.in/screens/1263243/nifty-midcap-100/",
    "smallcap_100": "https://www.screener.in/screens/1263244/nifty-smallcap-100/",

    # Sector-based scopes
    "auto_sector": "https://www.screener.in/screens/1263245/auto-sector/",
    "banking_sector": "https://www.screener.in/screens/1263246/banking-sector/",
    "it_sector": "https://www.screener.in/screens/1263247/it-sector/",
    "fmcg_sector": "https://www.screener.in/screens/1263248/fmcg-sector/",
    "pharma_sector": "https://www.screener.in/screens/1263249/pharma-sector/",
    "metal_sector": "https://www.screener.in/screens/1263250/metal-sector/",
    "oil_gas_sector": "https://www.screener.in/screens/1263251/oil-and-gas-sector/",
    "realty_sector": "https://www.screener.in/screens/1263252/realty-sector/",
    "energy_sector": "https://www.screener.in/screens/1263253/energy-sector/",
    "infra_sector": "https://www.screener.in/screens/1263254/infrastructure-sector/",
    "chemicals_sector": "https://www.screener.in/screens/1263255/chemicals-sector/",
}
