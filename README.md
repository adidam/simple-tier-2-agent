# simple-tier-2-agent

A minimal, provider-agnostic agent that answers questions about stock valuation, price history, business/qualitative context, and current news using live data — with a planner+executor architecture, human-approved ticker correction, and now web search for qualitative questions the structured tools can't answer.

Progression so far:

- **Tier 2**: single LLM → tool → LLM round trip
- **Tier 2.5**: multiple tools, multi-tool-call handling in one response
- **Tier 3**: ReAct loop — call a tool, observe, decide whether to call another
- **Tier 4**: explicit planner + executor, with plan-vs-execution deviation checking
- **v4.1**: fixed plan instability via a few-shot example in the planner prompt
- **v4.2**: human-in-the-loop ticker correction with per-run caching
- **v4.3 (current)**: added a fourth tool, `get_web_search` (DuckDuckGo, no API key), for news/commentary/qualitative questions; both system prompts updated to describe all four tools and to source web-search claims explicitly

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "What is the latest news on IDFCFIRSTB.NS?"
uv run python agent.py "Give me a full picture of INFY.NS: valuation, past year performance, what its business does, and how it compares to TCS.NS"
```

The second example is the tier-4.3 test case: a question only `get_web_search` can answer, since none of the other three tools carry current events or commentary.

The agent:

1. **Plans** — a dedicated call (no tools attached) produces a structured JSON plan. The planner is explicitly told to only include a `get_web_search` step when the question genuinely needs current news or qualitative judgment — not reflexively on every plan.
2. **Executes** — the plan is injected as context into the ReAct loop. If a tool call returns a "not found" style error, the model can suggest a corrected ticker, gated behind human approval (cached per run, so the same ticker isn't re-prompted per tool).
3. **Checks for deviation** — planned calls are compared against executed calls, treating an approved substitution as fulfilling the original planned step.
4. **Answers** — facts are separated from interpretation, and web-search-derived claims are explicitly attributed ("according to recent coverage") rather than stated with the same certainty as structured financial data.

## Tools

- `get_stock_info(ticker)` — current price and financial ratios
- `get_price_history(ticker, period)` — historical price performance
- `get_company_profile(ticker)` — business description, sector, industry, employee count
- `get_web_search(query)` — DuckDuckGo web search (unofficial library, no API key) for news, commentary, and qualitative context

## Tech stack

- **Python**, managed with **uv**
- **yfinance** — structured market data
- **ddgs** (DuckDuckGo search, unofficial) — web search, chosen deliberately over a polished paid API (e.g. Tavily) to build and own the failure-handling rather than depend on a provider that handles it invisibly
- **OpenAI SDK** pointed at **OpenRouter** — provider-agnostic, model swappable via a single string
- **LangSmith** — tracing, with `@traceable` on the main loop

## Setup

```bash
uv init --no-workspace
uv add -r requirements.txt   # or: uv add openai python-dotenv yfinance langsmith ddgs
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
web_search_tool.py   # get_web_search — DuckDuckGo search, no API key
agent.py     # tool schemas, planner + executor, ticker-correction flow, deviation check
.env         # API keys (not committed)
pyproject.toml / uv.lock
```

Note: a `tools/` package restructuring (splitting each tool into its own file, plus a shared `common.py` for internal helpers) was explored on a `refactor/tools-package` branch and deliberately not merged — the web-search tool addition continued on `main` in the flat-file layout instead. Revisit the package restructuring later if the flat files become unwieldy again.

## Design notes

- All tools return trimmed payloads with a consistent `error` field, including on paths where the underlying library returns empty data rather than raising (a real bug found and fixed in `get_company_profile`).
- **Dependency test in the planner prompt**: a step depends on another ONLY if its arguments cannot be determined without the earlier step's result.
- **Plan instability, found and fixed (v4.1)**: identical questions initially produced plans of varying thoroughness on multi-entity comparisons. Fixed with a few-shot example demonstrating symmetric tool coverage.
- **Ticker correction is human-gated by design (v4.2)**: the model can propose a corrected ticker, but never applies it without explicit approval; corrections are cached per run to avoid re-prompting per tool call; declining stops the run rather than silently retrying.
- **Web search is unofficial and deliberately so (v4.3)**: `ddgs` has no formal API contract and can rate-limit or change behavior without notice. The tool's `error` field is treated as load-bearing rather than defensive boilerplate, since this tool fails more often in practice than the yfinance-backed ones. Chosen over a paid, purpose-built option (Tavily) specifically to build and understand failure-handling directly rather than depend on a provider that abstracts it away.
- **Both the planner and executor prompts list all four tools explicitly** — a lesson carried over from earlier tools: the model can sometimes find and use a tool from its schema alone, but an incomplete system prompt makes correct, consistent tool selection more fragile, especially as the tool count grows.
- **The planner is explicitly told not to add a web-search step reflexively** — verified in both directions: a plain valuation question produces no web-search step, and a genuinely qualitative/news question does.

## Roadmap

- [x] Second tool (price history)
- [x] Multiple simultaneous tool calls in one response
- [x] ReAct loop for multi-step, dependent questions
- [x] Third tool (company profile) for qualitative business questions
- [x] Planner + executor with plan-vs-execution deviation checking
- [x] Plan instability investigated and fixed via few-shot example
- [x] Human-in-the-loop ticker correction, with caching and deviation-checker reconciliation
- [x] Fourth tool: web search (DuckDuckGo) for news/qualitative questions, with source attribution in answers
- [ ] Revisit tools/ package restructuring if flat files become unwieldy
- [ ] Multi-agent architecture (tier 5, likely via LangGraph) as the system grows
- [ ] Standalone RAG project (10-Ks, earnings transcripts) before merging back in as a tool
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
