import yfinance as yf
from ._common import _fetch_url, _safe_get
from typing import Dict, Any, Optional


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
            import yfinance as yf

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


def suggest_ticker_correction(bad_ticker: str) -> str | None:
    # strip suffix/noise, search the company name portion
    search_term = bad_ticker.replace(".NS", "").replace(".BO", "")
    try:
        results = yf.Search(search_term, max_results=3).quotes
        for r in results:
            if r.get("exchange") in ("NSI", "BSE"):  # prefer Indian exchanges
                return r["symbol"]
        return results[0]["symbol"] if results else None
    except Exception:
        return None
