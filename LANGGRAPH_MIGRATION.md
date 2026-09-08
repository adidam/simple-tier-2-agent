## LangGraph migration (in progress, separate file)

`agent_graph.py` is a parallel, standalone re-implementation of the tier-3 ReAct loop
(4 tools, no planner yet) using LangGraph instead of the hand-rolled `for` loop in
`agent.py`. It exists alongside `agent.py`, not as a replacement — the migration is
being done incrementally, verified against the same test cases at each step.

Confirmed working:

- State (`AgentState` with `add_messages`) replaces the hand-maintained `messages` list
- `ToolNode` (prebuilt) replaces the hand-written `dispatch_tool` if/elif chain, and
  handles multiple simultaneous tool calls automatically
- `tools_condition` (prebuilt) replaces the `if not msg.tool_calls: return` check
- LangSmith tracing works automatically for LangGraph runs (`ls_integration: langgraph`)
  with no `@traceable` decorator or `wrap_openai` wrapper needed — unlike the raw
  OpenAI SDK version, which required both

### 🚨 Confirmed bug, found via stress-testing: hallucination on unhandled tool failure

`agent_graph.py` does NOT yet have the ticker-correction / stop-on-decline flow that
`agent.py` has (v4.2). Tested with the same bad ticker (`IDFCFIRST.NS`) that `agent.py`
correctly handles by stopping and telling the user the ticker wasn't found:

- `agent.py` (with the correction flow): stops cleanly, tells the user the data
  couldn't be retrieved, never invents numbers.
- `agent_graph.py` (missing the correction flow): after the tool 404'd twice, the model
  produced a full financial comparison prefaced with "Based on my research," containing
  **fabricated figures** — e.g. reported IDFC First Bank's ROE as 3.4%, versus the
  actual tool-sourced value of ~47.7% confirmed earlier via `agent.py` on the correct
  ticker (IDFCFIRSTB.NS). It also presented metrics (NIM, ROCE, Revenue TTM) that don't
  exist anywhere in `get_stock_info`'s actual output schema, as if they came from tool
  data.

This is a direct violation of the executor system prompt's explicit instruction to
"never invent or guess financial data" — occurring specifically because the tool-failure
safety net (human-gated ticker correction, stop-on-decline) built in v4.2 was not yet
ported to the LangGraph version. Confirms that safety net is load-bearing, not
cosmetic: removing it (even via an incomplete migration, not a deliberate choice)
reproduces exactly the failure mode it was built to prevent.

**Not yet fixed. Next steps, in order:**

1. Port the human-in-the-loop ticker-correction + stop-on-decline flow into
   `agent_graph.py`, using LangGraph's native `interrupt()` primitive instead of the
   raw `input()` call used in `agent.py`.
2. Consider strengthening the system prompt further: explicit instruction to refuse
   to answer (not just avoid inventing specific numbers) when a required tool fails
   and cannot be resolved, rather than silently proceeding with unstated/trained
   knowledge as if it were tool output.
3. Re-run this same bad-ticker test after the fix, confirm it now stops cleanly like
   `agent.py` does.
4. THEN proceed to the originally planned deliberate weak-model stress test
   (e.g. via `openrouter/free` or a specific cheap paid model), now that a known gap
   isn't confounding the results.

### Also not yet migrated

- The planner node (tier 4) — agent_graph.py currently only reproduces the tier-3 loop
- The plan-vs-execution deviation checker

### Model/cost notes from this session

- `openrouter/auto` silently routed to a $0-cost model at one point, with no prompt
  caching involved (`cache_read: 0` throughout) — confirmed via OpenRouter's own
  activity log showing zero cost per request, not a caching discount.
- Pinning `ibm/granite-4.0-micro` directly failed with a 404 ("no endpoints found that
  support tool use") — the base model supports tool calling per IBM's model card, but
  the specific OpenRouter-hosted endpoint for it did not expose that capability. Lesson:
  model-level capability and endpoint-level capability on OpenRouter can diverge.
- `openrouter/free` is a router endpoint that filters candidate free models by required
  capability (e.g. tool calling) before random selection — safer than pinning an
  arbitrary free model ID directly, at the cost of not knowing in advance which
  underlying model will actually serve a given request.
