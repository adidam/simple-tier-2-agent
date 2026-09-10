from tools import (
    get_stock_info as _get_stock_info,
    get_price_history as _get_price_history,
    get_company_profile as _get_company_profile,
    get_web_search as _get_web_search,
)
from agent import is_not_found_error, suggest_ticker_correction
from langchain_core.tools import tool
from langgraph.types import interrupt
from typing import Literal

# module-level, reset per process — good enough for a CLI run
_ticker_corrections: dict[str, str] = {}


def handle_ticker_error(get_data_fn, ticker: str, tool_name: str) -> dict:
    """Handle errors for ticker-based data retrieval."""

    # if this ticker was already corrected earlier in this run, reuse it silently
    if ticker in _ticker_corrections:
        return get_data_fn(_ticker_corrections[ticker])

    result = get_data_fn(ticker)
    if not is_not_found_error(result):
        return result

    suggestion = suggest_ticker_correction(ticker)
    answer = interrupt({
        "type": "ticker_correction",
        "tool": tool_name,
        "original_ticker": ticker,
        "suggested_ticker": suggestion,
    }).strip()

    if answer.lower() == "y" and suggestion:
        _ticker_corrections[ticker] = suggestion
        return get_data_fn(suggestion)
    elif answer.lower() == "n" or not answer:
        return {"error": f"Stopped: '{ticker}' not found, correction declined."}
    else:
        _ticker_corrections[ticker] = answer.upper()
        return get_data_fn(answer.upper())


# ── 2. TOOLS ──────────────────────────────────────────────
# @tool reads each function's docstring + type hints and builds the JSON
# schema automatically — this replaces the hand-written schema dicts you
# maintained by hand in agent.py's `tools` list.


@tool
def get_stock_info(ticker: str) -> dict:
    """Get current price and key financial ratios (PE, market cap, etc.) for a stock ticker (NSE tickers end in .NS)."""
    return handle_ticker_error(_get_stock_info, ticker, "get_stock_info")


@tool
def get_price_history(
    ticker: str,
    period: Literal["1d", "5d", "1mo", "3mo", "6mo",
                    "1y", "2y", "5y", "10y", "ytd", "max"] = "6mo"
) -> dict:
    """Fetch historical price performance for a stock ticker over a specified period."""
    return handle_ticker_error(lambda t: _get_price_history(t, period), ticker, "get_price_history")


@tool
def get_company_profile(ticker: str) -> dict:
    """Get a company's business description, sector, industry, and employee count. Use for 'what does this business do' questions, NOT price or valuation."""
    return handle_ticker_error(_get_company_profile, ticker, "get_company_profile")


@tool
def get_web_search(query: str) -> dict:
    """Search the web for current news, commentary, or qualitative context not available from the other tools (e.g. management changes, recent events, analyst sentiment)."""
    return handle_ticker_error(_get_web_search, query, "get_web_search")
