import os
from tools import (
    get_stock_info as _get_stock_info,
    get_price_history as _get_price_history,
    get_company_profile as _get_company_profile,
    get_web_search as _get_web_search,
)

from openai import OpenAI
from langsmith.wrappers import wrap_openai

from langchain_core.tools import tool
from langgraph.types import interrupt
from typing import Literal

from tools import suggest_ticker_correction_yf

# module-level, reset per process — good enough for a CLI run
_ticker_corrections: dict[str, str | None] = {}


def is_not_found_error(result: dict) -> bool:
    err = (result.get("error") or "").lower()
    return "not found" in err or "no profile data" in err or "unable to fetch" in err


def suggest_ticker_correction(bad_ticker: str) -> str | None:
    # for ticker correction suggestions only, not the main executor
    CHEAP_MODEL = "meta-llama/llama-3.1-8b-instruct"
    client = wrap_openai(OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ))

    try:
        response = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a ticker-symbol correction assistant. Given a stock ticker that "
                    "returned no data, suggest the most likely correct NSE ticker symbol "
                    "(ending in .NS). Respond with ONLY the ticker, or the single word UNKNOWN "
                    "if you have no confident guess. No explanation."
                )},
                {"role": "user", "content": f"This ticker returned no data: {bad_ticker}"}
            ])
        suggestion = response.choices[0].message.content.strip()
        return None if suggestion.upper() == "UNKNOWN" else suggestion
    except Exception as e:
        print(
            f"⚠️ Correction suggestion unavailable ({e}); proceeding without a suggested fix.")
        return None


def handle_ticker_error(get_data_fn, ticker: str, tool_name: str) -> dict:
    """Handle errors for ticker-based data retrieval."""

    # if this ticker was already corrected earlier in this run, reuse it silently
    if ticker in _ticker_corrections:
        corrected = _ticker_corrections[ticker]
        if corrected is None:
            return {"error": f"Ticker '{ticker}' was not found and correction was declined."}
        return get_data_fn(corrected)

    result = get_data_fn(ticker)
    if not is_not_found_error(result):
        return result

    suggestion = suggest_ticker_correction_yf(ticker)
    if suggestion is None:
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
        # Cache None so we know not to interrupt again in this session
        _ticker_corrections[ticker] = None
        return {"error": f"Stopped: '{ticker}' not found, correction declined."}
    else:
        _ticker_corrections[ticker] = answer.upper()
        return get_data_fn(answer.upper())


# ── 2. TOOLS ──────────────────────────────────────────────
# @tool reads each function's docstring + type hints and builds the JSON
# schema automatically — this replaces the hand-written schema dicts you
# maintained by hand in agent.py's `tools` list.

@tool
def lookup_stock_symbol(company_name_or_query: str) -> dict:
    """Look up a stock ticker symbol by company name or query (e.g. 'Infosys', 'Reliance Industries')."""
    return _get_web_search(company_name_or_query)


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
    return _get_web_search(query)
