"""Small utility to fetch stock price and key ratios.

Uses Yahoo Finance unofficial JSON endpoints with a fallback to yfinance if available.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

try:
    import requests
except Exception:  # pragma: no cover - best-effort import
    requests = None  # type: ignore

import urllib.request
import urllib.error
import yfinance as yf


def _fetch_url(url: str, timeout: int = 10, headers: Optional[Dict[str, str]] = None):
    """Fetch a URL returning (status_code, text, json_or_None).

    Uses `requests` if available, otherwise `urllib` from the stdlib.
    """
    headers = headers or {}
    if requests is not None:
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            text = resp.text
            status = getattr(resp, "status_code", None)
            try:
                j = resp.json()
            except Exception:
                j = None
            return status, text, j
        except Exception:
            return None, "", None
    # urllib fallback
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            text = raw.decode("utf-8", errors="replace")
            status = getattr(r, "status", None)
            try:
                j = json.loads(text)
            except Exception:
                j = None
            return status, text, j
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            text = str(e)
        return getattr(e, "code", None), text, None
    except Exception:
        return None, "", None


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def get_stock_info(symbol: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch latest price and some key ratios for `symbol`.

    Uses `yfinance` library as primary method (recommended).
    Falls back to Yahoo Finance JSON endpoints if yfinance fails.

    For Indian stocks without a suffix, tries .NS (NSE) suffix automatically.

    Returns a dict with keys:
      - symbol: normalized ticker
      - price: float or None
      - currency: currency code or None
      - key_ratios: dict of ratio name -> value
      - raw: raw data when available
      - error: error message if fetching failed
    """
    symbol = symbol.strip().upper()
    result: Dict[str, Any] = {"symbol": symbol, "price": None,
                              "currency": None, "key_ratios": {}, "error": None}

    def _try_yfinance(ticker: str) -> Optional[Dict[str, Any]]:
        """Try to fetch data from yfinance for a given ticker."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            # Check if we got meaningful data (at least price or market cap)
            price = info.get("regularMarketPrice") or info.get(
                "currentPrice") or info.get("previousClose")
            market_cap = info.get("marketCap")

            if price is None and market_cap is None:
                # No meaningful data, try next option
                return None

            key_ratios = {
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": info.get("priceToBook"),
                "beta": info.get("beta"),
                "marketCap": info.get("marketCap"),
                "debtToEquity": info.get("debtToEquity"),
                "currentRatio": info.get("currentRatio"),
                "returnOnEquity": info.get("returnOnEquity"),
                "profitMargins": info.get("profitMargins"),
            }
            return {
                "price": price,
                "currency": info.get("currency"),
                "key_ratios": {k: v for k, v in key_ratios.items() if v is not None}
            }
        except Exception:
            pass
        return None

    # Primary: try yfinance with original symbol
    data = _try_yfinance(symbol)
    if data:
        result.update(data)
        return result

    # If no suffix and no data, try with .NS suffix (Indian NSE stocks)
    if "." not in symbol:
        data = _try_yfinance(f"{symbol}.NS")
        if data:
            result.update(data)
            return result
        # Also try .BO suffix (Indian BSE stocks)
        data = _try_yfinance(f"{symbol}.BO")
        if data:
            result.update(data)
            return result

    # Fallback: try Yahoo Finance direct endpoints (likely blocked, but worth a try)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
    try:
        status, text, data_direct = _fetch_url(
            url, timeout=timeout, headers=headers)
        if data_direct:
            quotes = data_direct.get("quoteResponse", {}).get("result") or []
            if quotes:
                q0 = quotes[0]
                result["price"] = q0.get("regularMarketPrice") or q0.get(
                    "regularMarketPreviousClose")
                result["currency"] = q0.get("currency")
                result["key_ratios"] = {
                    k: q0.get(k)
                    for k in ["marketCap", "beta", "trailingPE", "forwardPE", "priceToBook"]
                }
                result["key_ratios"] = {
                    k: v for k, v in result["key_ratios"].items() if v is not None}
                result["error"] = None
                return result
    except Exception:
        pass

    # If all else fails, return symbol only and error message
    if result["error"] is None:
        result["error"] = "Unable to fetch data from any source"
    return result


def get_price_history(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """Fetch price history for `symbol` over the given period.

    Uses yfinance's .history() method and returns a trimmed summary:
      - symbol: normalized ticker
      - period: requested period
      - start_price: opening price at the start of period
      - end_price: closing price at the end of period
      - percent_change: percentage change over period
      - high: highest price in the period
      - low: lowest price in the period
      - error: error message if fetching failed

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'TCS' or 'TCS.NS')
        period: yfinance period string ('1d', '5d', '1mo', '3mo', '6mo', '1y', etc.)
    """
    symbol = symbol.strip().upper()
    result: Dict[str, Any] = {
        "symbol": symbol,
        "period": period,
        "start_price": None,
        "end_price": None,
        "percent_change": None,
        "high": None,
        "low": None,
        "error": None
    }

    def _try_yfinance_history(ticker: str) -> Optional[Dict[str, Any]]:
        """Try to fetch history from yfinance for a given ticker."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)

            if hist is None or hist.empty:
                return None

            # Extract prices, skipping NaN values
            close_prices = hist['Close'].dropna()
            if len(close_prices) == 0:
                return None

            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            high = hist['High'].max()
            low = hist['Low'].min()

            # Calculate percent change
            percent_change = ((end_price - start_price) /
                              start_price * 100) if start_price != 0 else 0

            return {
                "start_price": float(start_price),
                "end_price": float(end_price),
                "percent_change": round(percent_change, 2),
                "high": float(high),
                "low": float(low)
            }
        except Exception:
            pass
        return None

    # Try with original symbol
    data = _try_yfinance_history(symbol)
    if data:
        result.update(data)
        return result

    # If no suffix and no data, try with .NS suffix (Indian NSE stocks)
    if "." not in symbol:
        data = _try_yfinance_history(f"{symbol}.NS")
        if data:
            result.update(data)
            return result
        # Also try .BO suffix (Indian BSE stocks)
        data = _try_yfinance_history(f"{symbol}.BO")
        if data:
            result.update(data)
            return result

    # If all else fails, return error
    if result["error"] is None:
        result["error"] = "Unable to fetch price history for the given symbol"
    return result


def get_company_profile(symbol: str) -> Dict[str, Any]:
    """Fetch company profile information for `symbol`.

    Uses yfinance's .info method to retrieve company details.

    Returns a dict with keys:
      - symbol: normalized ticker
    """
    symbol = symbol.strip().upper()
    result: Dict[str, Any] = {"symbol": symbol, "profile": {}, "error": None}

    try:
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


def _print_json(d: Dict[str, Any]) -> None:
    print(json.dumps(d, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools.py SYMBOL")
        sys.exit(1)
    sym = sys.argv[1]
    out = get_company_profile(sym)
    _print_json(out)
