from ._common import _fetch_url, _safe_get
from typing import Dict, Any, Optional


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
            import yfinance as yf

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
