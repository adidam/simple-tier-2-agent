# simple-tier-2-agent

A minimal, provider-agnostic tool-calling agent that answers questions about stock prices and financial ratios using live market data.

This is a **tier-2 agent**: a single LLM → tool → LLM round trip, no loop, no memory. It exists as a deliberately small, finished building block on the way to a larger agentic value-investing system.

## What it does

Ask a question like:

```
uv run python agent.py "What's the forward PE of INFY.NS?"
```

The agent:

1. Sends your question to an LLM (via OpenRouter) along with a `get_stock_info` tool definition
2. The model decides to call the tool with the ticker it inferred from your question
3. `tools.py` fetches live price and key ratios via `yfinance`
4. The result is sent back to the model, which produces a grounded natural-language answer

## Tech stack

- **Python**, managed with **uv** (not pip/venv)
- **yfinance** — market data source
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic LLM access (swap models via a single string, no code changes)
- **LangSmith** — optional tracing/observability for inspecting each tool-call round trip

## Setup

```bash
uv init --no-workspace
uv add -r requirements.txt   # or: uv add openai python-dotenv yfinance langsmith
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LANGSMITH_API_KEY=lsv2_pt_your-key-here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=simple-tier-2-agent
```

`.env` is git-ignored — never commit it. `uv.lock` **is** committed for reproducible installs.

## Usage

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python tools.py INFY.NS   # test the data function directly, no LLM involved
```

## Project structure

```
tools.py     # get_stock_info(ticker) — data layer, no LLM dependency
agent.py     # tool schema + single tool-call round trip
.env         # API keys (not committed)
pyproject.toml / uv.lock   # dependencies (uv-managed)
```

## Design notes

- `get_stock_info` returns a trimmed payload (`price`, `key_ratios`, `error`) — the full yfinance response is dropped before it reaches the model, to keep token usage and latency down.
- Invalid tickers return a clean `error` field instead of raising, so the failure path is predictable for both the agent and any caller.
- Model is set via a single `MODEL` string in `agent.py` — swappable across any model OpenRouter hosts.

## Roadmap

- [ ] Second tool (e.g. historical price / peer comparison) — teaches the model to choose between tools
- [ ] ReAct-style loop for multi-step questions
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
