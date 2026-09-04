# simple-tier-2-agent

A minimal, provider-agnostic agent that answers questions about stock valuation, price history, and business/qualitative context using live market data — with a planner+executor architecture for compound, multi-part questions.

Progression so far:

- **Tier 2**: single LLM → tool → LLM round trip
- **Tier 2.5**: multiple tools, multi-tool-call handling in one response
- **Tier 3**: ReAct loop — call a tool, observe, decide whether to call another
- **Tier 4 (current)**: explicit planner + executor — the model writes a plan before executing, then the executor loop runs against that plan, with a check that flags when execution deviates from what was planned

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "What does INFY.NS's business actually do?"
uv run python agent.py "Give me a full picture of INFY.NS: valuation, past year performance, what its business does, and how it compares to TCS.NS"
```

The last example is the tier-4 test case: a compound question spanning valuation, momentum, qualitative business context, and a peer comparison — the kind of question a flat ReAct loop can handle but tends to approach without an inspectable strategy.

The agent:

1. **Plans** — a dedicated call (no tools attached) asks the model to lay out its intended tool calls as structured JSON, including a `depends_on` field per step (only set when a step's arguments genuinely can't be known until an earlier step resolves — not just because two results will later be compared)
2. **Executes** — the plan is injected as prior context into the existing tier-3 ReAct loop, which then calls tools, observes results, and continues until it has enough to answer
3. **Checks for deviation** — after execution, planned tool calls (for steps with a known ticker) are compared against what was actually executed, and any planned-but-skipped step is flagged
4. **Answers** — the model produces a final synthesized answer from everything gathered

## Tools

- `get_stock_info(ticker)` — current price and financial ratios
- `get_price_history(ticker, period)` — historical price performance
- `get_company_profile(ticker)` — business description, sector, industry, employee count (added in tier 4 specifically to answer qualitative "what does this business do" questions that pure ratios can't address)

## Tech stack

- **Python**, managed with **uv**
- **yfinance** — market data source
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic, model swappable via a single string
- **LangSmith** — tracing, with `@traceable` on the main loop so a full multi-step question nests under one parent trace

## Setup

```bash
uv init --no-workspace
uv add -r requirements.txt   # or: uv add openai python-dotenv yfinance langsmith
```

`.env`:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LANGSMITH_API_KEY=lsv2_pt_your-key-here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=simple-tier-2-agent
```

`.env` is git-ignored. `uv.lock` is committed for reproducible installs.

## Project structure

```
tools.py     # get_stock_info, get_price_history, get_company_profile — data layer, no LLM dependency
agent.py     # tool schemas, planner prompt + call, executor loop, deviation check
.env         # API keys (not committed)
pyproject.toml / uv.lock
```

## Design notes

- All three tools return trimmed payloads with a consistent `error` field — including on paths where the underlying library doesn't raise an exception but returns empty data (a real bug found and fixed in `get_company_profile`, where an invalid ticker returned `error: null` alongside an empty profile until explicitly checked for).
- The planner uses a separate system prompt and is called with no `tools` parameter, forcing a text/JSON plan rather than a tool_calls response.
- **Dependency test in the planner prompt**: a step depends on another ONLY if its arguments cannot be determined without the earlier step's result — explicitly NOT just because two results will be compared or discussed together later. This distinction was loose in an early version of the prompt and had to be tightened; without it, the model over-reported dependencies that weren't real.
- The plan is injected into the executor's message history as a prior assistant turn, so the existing tier-3 loop (unchanged) tends to act on it without needing a rewritten execution engine.
- **Known limitation — plan instability**: the same question, run twice, can produce different plans (e.g., one run planned 6 steps including a peer's business profile; a later run planned only 5 and omitted it). This is model non-determinism in the planning step itself, distinct from execution deviating from a fixed plan. Not solved here — noted as a real, observed limitation of the pattern rather than an assumed one.
- **Deviation checking**: after execution, planned steps with a known ticker are compared against what actually ran; a mismatch prints a warning. Steps with `ticker: null` (genuinely dependent steps, resolved only at runtime) are skipped in this comparison, since they can't be matched by ticker ahead of time — a harder problem left open.

## Roadmap

- [x] Second tool (price history)
- [x] Multiple simultaneous tool calls in one response
- [x] ReAct loop for multi-step, dependent questions
- [x] Third tool (company profile) for qualitative business questions
- [x] Planner + executor with plan-vs-execution deviation checking
- [ ] Address plan instability (e.g. lower planner temperature, few-shot examples, or a dedicated planning model)
- [ ] Web search tool for questions with no structured data source (e.g. management quality, recent news)
- [ ] Multi-agent architecture (likely via LangGraph) as the system grows toward the full value-investing use case
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
