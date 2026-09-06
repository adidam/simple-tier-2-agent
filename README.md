# simple-tier-2-agent

A minimal, provider-agnostic agent that answers questions about stock valuation, price history, and business/qualitative context using live market data — with a planner+executor architecture for compound, multi-part questions.

Progression so far:

- **Tier 2**: single LLM → tool → LLM round trip
- **Tier 2.5**: multiple tools, multi-tool-call handling in one response
- **Tier 3**: ReAct loop — call a tool, observe, decide whether to call another
- **Tier 4**: explicit planner + executor — the model writes a plan before executing, then the executor loop runs against that plan, with a check that flags when execution deviates from what was planned
- **v4.1 (current)**: fixed plan instability with a few-shot example in the planner prompt; tightened the executor prompt to cover all three tools, explain the injected plan, and separate fact from interpretation

## What it does

```bash
uv run python agent.py "What's the forward PE of INFY.NS?"
uv run python agent.py "What does INFY.NS's business actually do?"
uv run python agent.py "Give me a full picture of INFY.NS: valuation, past year performance, what its business does, and how it compares to TCS.NS"
```

The last example is the tier-4 test case: a compound question spanning valuation, momentum, qualitative business context, and a peer comparison.

The agent:

1. **Plans** — a dedicated call (no tools attached) asks the model to lay out its intended tool calls as structured JSON, including a `depends_on` field per step (only set when a step's arguments genuinely can't be known until an earlier step resolves — not just because two results will later be compared). The prompt includes a worked few-shot example showing that comparison questions require the SAME depth of tool calls for EACH entity being compared, not just the one named first in the question.
2. **Executes** — the plan is injected as prior context into the existing tier-3 ReAct loop. The executor's system prompt explains that the plan is a strong guide (not a rigid script), names all three available tools, and instructs the model to explicitly report any planned step it decides to skip and why.
3. **Checks for deviation** — after execution, planned tool calls (for steps with a known ticker) are compared in code against what was actually executed, and any planned-but-skipped step is flagged. This is a second, independent check alongside the model's own self-reported plan adherence.
4. **Answers** — the model produces a final synthesized answer, explicitly separating factual data from its own interpretation (e.g. moat or competitive-position judgments).

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
agent.py     # tool schemas, planner prompt + call, executor loop, deviation check
.env         # API keys (not committed)
pyproject.toml / uv.lock
```

## Design notes

- All three tools return trimmed payloads with a consistent `error` field — including on paths where the underlying library doesn't raise an exception but returns empty data (a real bug found and fixed in `get_company_profile`).
- The planner uses a separate system prompt and is called with no `tools` parameter, forcing a JSON plan rather than a tool_calls response.
- **Dependency test in the planner prompt**: a step depends on another ONLY if its arguments cannot be determined without the earlier step's result — explicitly NOT just because two results will be compared later.
- **Plan instability, found and fixed**: the same compound question, run twice, initially produced different plans (a 6-step plan covering both compared companies fully vs. a 5-step plan that silently dropped one company's profile lookup). Fixed by adding a few-shot example to the planner prompt showing that comparison questions require symmetric tool coverage across every entity mentioned. Confirmed stable across multiple repeated runs, including on a different pair of tickers, after the fix.
- **Executor prompt now explicitly**: names all three tools, explains that an injected plan is a guide (not a script) and any skipped step must be announced, and instructs fact/interpretation separation for qualitative topics. Previously this prompt only mentioned two tools and said nothing about the plan or interpretation — the model was still finding and using the third tool via its schema alone, but silent step-skipping and unflagged interpretive claims were more likely without explicit instruction.
- **Observed: model self-correction on bad tickers.** When a planned ticker 404'd (e.g. `IDFCFIRST.NS`), the model inferred and retried the correct real ticker (`IDFCFIRSTB.NS`) on its own and disclosed the substitution in its answer. Useful, but noted as an open design question: this relies on the model guessing correctly, and a similarly-named-but-different company could plausibly be substituted incorrectly. Not yet decided whether to keep this autonomy or force a stop-and-ask on ticker-not-found instead.
- **Known gap: the code-level deviation checker matches by exact ticker string**, so a model self-correction like the one above (different ticker than planned, same intent) would likely be flagged as a false-positive deviation by the code checker even though the model's own self-report correctly distinguished "all steps completed" from "one ticker substituted." The two checks (model self-report vs. code-level check) can disagree on cases like this — not yet reconciled.

## Roadmap

- [x] Second tool (price history)
- [x] Multiple simultaneous tool calls in one response
- [x] ReAct loop for multi-step, dependent questions
- [x] Third tool (company profile) for qualitative business questions
- [x] Planner + executor with plan-vs-execution deviation checking
- [x] Plan instability investigated and fixed via few-shot example, confirmed across repeated runs
- [ ] Reconcile the code-level deviation checker with legitimate ticker substitutions (currently a likely false positive)
- [ ] Decide whether to allow model-driven ticker self-correction or force explicit confirmation
- [ ] Web search tool for questions with no structured data source (e.g. management quality, recent news)
- [ ] Multi-agent architecture (likely via LangGraph) as the system grows toward the full value-investing use case
- [ ] Extend toward an agentic value-investing system (Zerodha/NSE data, paper trading, position guardrails)
