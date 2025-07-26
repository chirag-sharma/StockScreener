import requests
from bs4 import BeautifulSoup
from src.constants import SCOPE_URL_MAP


def fetch_tickers_from_screener(scope: str) -> list[str]:
    """
    Fetches the list of stock tickers for the given index or sector scope from Screener.in.

    Args:
        scope (str): One of the predefined keys in SCOPE_URL_MAP

    Returns:
        list[str]: List of stock tickers (symbols)

    Raises:
        ValueError: If the scope is not found in SCOPE_URL_MAP
        requests.RequestException: If there is a network error
    """
    if scope not in SCOPE_URL_MAP:
        raise ValueError(f"Invalid scope '{scope}'. Available options: {list(SCOPE_URL_MAP.keys())}")

    url = SCOPE_URL_MAP[scope]
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Send HTTP GET request to Screener.in
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse HTML response to extract ticker symbols
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="data-table")

    tickers = []
    if table:
        rows = table.find_all("tr")[1:]  # Skip header row
        for row in rows:
            cells = row.find_all("td")
            if cells:
                link = cells[0].find("a")
                if link and link.text:
                    tickers.append(link.text.strip())

    return tickers
