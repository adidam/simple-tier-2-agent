# simple-tier-2-agent

A minimal, provider-agnostic tool-calling agent that answers questions about stock prices, financial ratios, and price history using live market data.

This started as a **tier-2 agent** (a single LLM → tool → LLM round trip, no loop, no memory) and has grown to handle **multiple tools in one response** — the model can choose one or more tools per question, but there is still no loop: each question resolves in a single round trip.

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "How has INFY.NS performed over the last 6 months?"
uv run python agent.py "What's INFY.NS's current price and how has it moved over 6 months?"
```

The agent:

1. Sends your question to an LLM (via OpenRouter) along with two tool definitions: `get_stock_info` and `get_price_history`
2. The model decides which tool(s) to call — it may request one or both in the same response, depending on the question
3. `tools.py` fetches the relevant data via `yfinance`
4. All tool results are sent back to the model in one batch, which produces a single grounded final answer

## Tech stack

- **Python**, managed with **uv** (not pip/venv)
- **yfinance** — market data source
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic LLM access (swap models via a single string, no code changes)
- **LangSmith** — tracing/observability for inspecting each tool-call round trip

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
uv run python tools.py INFY.NS   # test get_stock_info directly, no LLM involved
```

## Project structure

```
tools.py     # get_stock_info(ticker), get_price_history(ticker, period) — data layer, no LLM dependency
agent.py     # tool schemas, system prompt, multi-tool-call round trip
.env         # API keys (not committed)
pyproject.toml / uv.lock   # dependencies (uv-managed)
```

## Design notes

- Both tools return trimmed payloads (e.g. `price`, `key_ratios`, `error` for `get_stock_info`) — the full yfinance response is never sent to the model, to keep token usage and latency down.
- Invalid tickers return a clean `error` field instead of raising, so the failure path is predictable for both the agent and any caller.
- A system prompt explicitly tells the model to report tool errors rather than invent data — this behavior is a deliberate instruction, not left to the model's default judgment.
- **Multiple tool calls in one response**: with two tools available, the model can request both in a single turn (e.g., "current price and 6-month history" triggers both `get_stock_info` and `get_price_history` together). The code loops over every entry in `tool_calls`, dispatches each by tool name, and sends back one `"role": "tool"` result per call — matched via `tool_call_id` — before making the final request. This is still one round trip, not a loop across multiple turns.
- Tool execution is wrapped in `try/except` separately from JSON-argument parsing, so a malformed tool call and a runtime failure inside the tool itself are handled distinctly rather than both crashing the script.
- Model is set via a single `MODEL` string in `agent.py` — swappable across any model OpenRouter hosts.

## Roadmap

- [x] Second tool (price history) — teaches the model to choose between tools
- [x] Handle multiple simultaneous tool calls in one response
- [ ] ReAct-style loop for multi-step questions (call a tool, observe, decide to call another)
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
