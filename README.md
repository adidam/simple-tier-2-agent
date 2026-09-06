# simple-tier-2-agent

A minimal, provider-agnostic agent that answers questions about stock valuation, price history, and business/qualitative context using live market data — with a planner+executor architecture for compound, multi-part questions and a human-approved ticker-correction flow.

Progression so far:

- **Tier 2**: single LLM → tool → LLM round trip
- **Tier 2.5**: multiple tools, multi-tool-call handling in one response
- **Tier 3**: ReAct loop — call a tool, observe, decide whether to call another
- **Tier 4**: explicit planner + executor, with plan-vs-execution deviation checking
- **v4.1**: fixed plan instability via a few-shot example in the planner prompt; tightened the executor prompt
- **v4.2 (current)**: human-in-the-loop ticker correction — the model can suggest a fix for a bad ticker, but a human must approve it before it's used; corrections are cached per run so the same ticker isn't re-prompted on every tool call; the deviation checker now recognizes an approved substitution as fulfilling the plan

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "Give me a full picture of INFY.NS: valuation, past year performance, what its business does, and how it compares to TCS.NS"
uv run python agent.py "Give me a full picture of UJJIVANSFB.NS ... and how it compares to IDFCFIRST.NS"
```

The last example deliberately uses a slightly wrong ticker (`IDFCFIRST.NS` instead of the real `IDFCFIRSTB.NS`) to exercise the correction flow — the agent detects the 404, asks whether to use the model's suggested correction, and either proceeds with your approval or stops cleanly if declined.

The agent:

1. **Plans** — a dedicated call (no tools attached) produces a structured JSON plan, including a `depends_on` field per step and, for multi-entity comparisons, symmetric tool coverage across every entity (fixed in v4.1 after finding the planner would sometimes cover one company more thoroughly than another).
2. **Executes** — the plan is injected as context into the ReAct loop from tier 3. If a tool call returns a "not found" style error, the model is asked to suggest a corrected ticker; a human is prompted to accept or decline before it's used. Accepted corrections are cached for the rest of that run.
3. **Checks for deviation** — planned tool calls are compared against what was executed, treating an approved substitution as covering the original planned step rather than flagging it as skipped.
4. **Answers** — the model produces a final answer, separating factual data from its own interpretation, and discloses any ticker substitution that occurred.

## Tools

- `get_stock_info(ticker)` — current price and financial ratios
- `get_price_history(ticker, period)` — historical price performance
- `get_company_profile(ticker)` — business description, sector, industry, employee count

## Tech stack

- **Python**, managed with **uv**
- **yfinance** — market data source
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic, model swappable via a single string
- **LangSmith** — tracing, with `@traceable` on the main loop

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
agent.py     # tool schemas, planner + executor, ticker-correction flow, deviation check
.env         # API keys (not committed)
pyproject.toml / uv.lock
```

## Design notes

- All three tools return trimmed payloads with a consistent `error` field, including on paths where the underlying library returns empty data rather than raising.
- **Dependency test in the planner prompt**: a step depends on another ONLY if its arguments cannot be determined without the earlier step's result.
- **Plan instability, found and fixed (v4.1)**: identical questions initially produced plans of varying thoroughness (e.g. dropping one company's profile lookup in a comparison). Fixed with a few-shot example demonstrating symmetric coverage; confirmed stable across repeated runs on multiple ticker pairs.
- **Ticker correction is human-gated by design (v4.2)**: the model can propose a corrected ticker when one 404s, but never applies it without an explicit `y`/`n` prompt — a deliberate choice over silent auto-correction, since a wrong guess (a similarly-named but different company) would otherwise go unnoticed.
- **Correction caching**: without caching, a single bad ticker used across multiple tools (info, history, profile) triggered the same prompt three times in one run, and declining on a later prompt could wipe out an already-approved earlier correction. Corrections are now cached per `ask()` call so the human is asked once per bad ticker, not once per tool call.
- **Declining a correction stops the run immediately** rather than letting the loop retry the same unresolved error — this was an actual infinite-loop-shaped bug (falling through with no `return` on decline) before being fixed.
- **Deviation checker updated to recognize substitutions**: matches planned calls against both the ticker actually used and any ticker it was substituted from, so an approved correction isn't misreported as a skipped step.
- Model is set via a single `MODEL` string in `agent.py` — swappable across any model OpenRouter hosts.

## Roadmap

- [x] Second tool (price history)
- [x] Multiple simultaneous tool calls in one response
- [x] ReAct loop for multi-step, dependent questions
- [x] Third tool (company profile) for qualitative business questions
- [x] Planner + executor with plan-vs-execution deviation checking
- [x] Plan instability investigated and fixed via few-shot example
- [x] Human-in-the-loop ticker correction, with caching and a fixed deviation-checker/substitution reconciliation
- [ ] Web search tool for questions with no structured data source (e.g. management quality, recent news)
- [ ] Multi-agent architecture (likely via LangGraph) as the system grows toward the full value-investing use case
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
