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

### Update: ticker-correction interrupt() ported and verified

The hallucination bug above is fixed. Ported the ticker-correction + stop-on-decline
flow into `agent_graph.py` using LangGraph's native `interrupt()` — placed INSIDE each
tool wrapper function (stock_info, price_history, company_profile) rather than in a
custom replacement for ToolNode, since interrupt() pauses the whole graph regardless
of which function calls it. This let ToolNode stay prebuilt/unchanged, and keeps each
tool's correction logic colocated with the tool itself rather than centralized in a
generic node that would need to know about every tool's failure modes.

Requires an `InMemorySaver` checkpointer (state must persist across the pause) and a
`thread_id` in config; `ask()` now loops on `"__interrupt__" in result`, prompting the
user and resuming via `Command(resume=answer)` — replacing the old blocking `input()`
call site from the pre-interrupt version.

Re-tested with the exact original failure case (`IDFCFIRST.NS`, not just the correct
ticker): execution now correctly pauses, prompts, and either retries with the
corrected ticker or stops cleanly — no hallucinated figures. Confirmed fixed.

### New findings from re-testing (both worth fixing, neither blocking)

**1. Interrupt resume re-executes the whole node, not just the paused call.**
When a tool node contains multiple tool calls in one batch (e.g. stock_info +
price_history + company_profile all requested together with a bad ticker), resuming
from the FIRST interrupt causes LangGraph to replay the entire node from the top —
re-triggering the NEXT tool's interrupt fresh, producing repeated 404s for what
appeared to be the same already-answered correction. Observed: answered `y` once for
IDFCFIRST.NS, but the same ticker 404'd twice more afterward before the next distinct
interrupt (UJJIVAN.SI) appeared. Harmless here since all tools are read-only GETs, but
flagged as a real risk for any future tool with a side effect (e.g. placing a trade) —
resume-on-interrupt would double-execute it. Not fixed; noted for when this agent
eventually touches the Zerodha paper-trading system.

**2. `price_history`'s `period` argument has no value constraint in the LangGraph
version.** The model passed a literal date ('2024-01-01') instead of a valid period
string, causing a tool-side error ("Period '2024-01-01' is invalid, must be one of:
1d, 5d, 1mo..."). The original agent.py's hand-written JSON schema didn't have an enum
either, but the @tool-decorated version here relies only on a docstring example, which
the model didn't reliably follow. Fix: use `Literal["1d", "5d", "1mo", ...]` as the
type hint, which @tool converts into a real schema-level enum, making an invalid value
impossible to send rather than just discouraged in prose. Not yet applied.

### More serious finding: model overrode CORRECT tool data with a worse guess

Confirmed via LangSmith trace inspection: `company_profile` was called successfully
for IDFCFIRSTB.NS and returned the correct `fullTimeEmployees: 43059` in its tool
result. Despite this being present and correct in context, the final answer's
"Overview" table reported employee count as **"~30,000+ (estimated)"** — a materially
wrong number, self-generated rather than read from the successful tool call, with a
hedge word ("estimated") that makes it look like appropriate caution rather than what
it actually is: silently disregarding correct, retrieved data.

This is a more serious and harder-to-catch failure mode than the original hallucination
bug. That bug occurred on tool FAILURE (a recognizable, catchable case, now fixed via
the interrupt() correction flow). This occurs on tool SUCCESS — the retrieval, sourcing
instructions, and fact/interpretation split can all be working exactly as designed, and
a specific field can still be silently wrong, because the failure is in SYNTHESIS, not
retrieval: the model apparently treated the answer's "Overview" table as general
scene-setting prose (written from training-data recall) rather than a place requiring
the same grounding discipline applied to the numbers table later in the same response.

**Not yet fixed.** Candidate directions, not yet decided/implemented:

1. Strengthen the system prompt: explicit instruction that EVERY specific figure in
   the response, not just clearly "financial" ones, must trace to a tool call result —
   including seemingly minor overview/background details like employee count or
   founding year — and that hedge words ("estimated", "approximately") are not a
   substitute for actually checking against retrieved data when a tool result for that
   exact field exists in context.
2. Consider whether letting the model draft a big-picture "overview" preamble at all
   (as opposed to a purely tool-grounded facts table) is structurally risky — an
   overview framed as "what I already know about this company" invites recall over
   retrieval, even when retrieval is available and correct.
3. This is a genuine, open reliability question worth carrying into any future
   evaluation/eval-harness work (tier 6 territory) rather than something a single
   prompt tweak is likely to fully close.
