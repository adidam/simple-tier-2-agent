from ._common import _fetch_url, _safe_get
from typing import Dict, Any, Optional


def get_company_profile(symbol: str) -> Dict[str, Any]:
    """Fetch company profile information for `symbol`.

    Uses yfinance's .info method to retrieve company details.

    Returns a dict with keys:
      - symbol: normalized ticker
    """
    symbol = symbol.strip().upper()
    result: Dict[str, Any] = {"symbol": symbol, "profile": {}, "error": None}

    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        info = t.info or {}
        profile_keys = ["longBusinessSummary", "sector",
                        "industry", "website", "fullTimeEmployees"]
        profile = {k: info.get(k)
                   for k in profile_keys if info.get(k) is not None}
        result["profile"] = profile
        if not profile:
            result["error"] = f"No profile data found for {symbol}"

    except Exception as e:
        result["error"] = f"Unable to fetch company profile: {e}"

    return result
