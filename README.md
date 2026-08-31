# simple-tier-2-agent

A minimal, provider-agnostic tool-calling agent that answers questions about stock prices, financial ratios, and price history using live market data — now with a ReAct-style loop for questions that require multi-step reasoning.

Started as a **tier-2 agent** (single LLM → tool → LLM round trip), then grew to handle **multiple tools in one response**, and now supports a **ReAct loop**: the model can call a tool, see the result, and decide to call another tool based on what it learned — repeating until it has enough to answer.

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "How has INFY.NS performed over the last 6 months?"
uv run python agent.py "Which of INFY.NS and TCS.NS has the lower forward PE, and what's that stock's 6-month price history?"
```

The last example is the real test case for the loop: the model must fetch info for both tickers, compare them, and _then_ decide which one to fetch price history for — a decision it can't make until it's seen the first result.

The agent:

1. Sends your question to an LLM (via OpenRouter) along with two tool definitions: `get_stock_info` and `get_price_history`
2. Loops: on each pass, the model either requests tool(s) or produces a final answer
3. `tools.py` executes whatever's requested via `yfinance`
4. Results are fed back in, and the loop continues until the model responds with no further tool calls (or a safety cap is hit)

## Tech stack

- **Python**, managed with **uv** (not pip/venv)
- **yfinance** — market data source
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic LLM access (swap models via a single string, no code changes)
- **LangSmith** — tracing/observability, with `@traceable` on the main loop so a question's full multi-step chain nests under one parent trace

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
uv run python agent.py "Which of INFY.NS and TCS.NS has the lower forward PE, and what's that stock's 6-month price history?"
uv run python tools.py INFY.NS   # test get_stock_info directly, no LLM involved
```

## Project structure

```
tools.py     # get_stock_info(ticker), get_price_history(ticker, period) — data layer, no LLM dependency
agent.py     # tool schemas, system prompt, ReAct loop with iteration cap
.env         # API keys (not committed)
pyproject.toml / uv.lock   # dependencies (uv-managed)
```

## Design notes

- Both tools return trimmed payloads — the full yfinance response is never sent to the model, to keep token usage and latency down.
- Invalid tickers return a clean `error` field instead of raising, so the failure path is predictable.
- A system prompt explicitly tells the model to report tool errors rather than invent data, and that it may call tools across multiple turns if a question requires sequential reasoning.
- **ReAct loop**: the round trip runs inside `for iteration in range(max_iterations)`. Each pass calls the model, checks `msg.tool_calls`; if empty, the loop returns the model's answer immediately. If tools were requested, every entry in `tool_calls` is executed and its result appended before looping again. Using `range(max_iterations)` both drives the loop and enforces the iteration cap in one construct, rather than a separate manual counter.
- **Iteration cap (max_iterations = 5)**: without this, a confused model or a bad tool result could loop indefinitely, burning API cost with no ceiling. If the cap is hit, the agent returns a "couldn't resolve this in time" message instead of continuing forever.
- **Multiple tool calls in one response** (still supported inside each loop pass): `tool_call.function.name` is dispatched to the right function, with JSON-parsing errors and tool execution errors caught separately.
- The main `ask()` function is wrapped in `@traceable` so a full multi-step question shows up in LangSmith as one parent trace with each loop iteration nested as a child run, instead of separate unrelated top-level traces.
- Model is set via a single `MODEL` string in `agent.py` — swappable across any model OpenRouter hosts.

## Roadmap

- [x] Second tool (price history) — teaches the model to choose between tools
- [x] Handle multiple simultaneous tool calls in one response
- [x] ReAct-style loop for multi-step, dependent questions
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
