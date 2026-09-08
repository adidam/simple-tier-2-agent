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
- Tested against a multi-part comparison question (IDFC First Bank vs Ujjivan SFB)
  exercising all four tools in one run; output quality matched or exceeded `agent.py`

Not yet migrated (next increments):

- The planner node (tier 4) — agent_graph.py currently only reproduces the tier-3 loop
- The human-in-the-loop ticker correction — currently a blocking `input()` in `agent.py`;
  the LangGraph-native equivalent is an `interrupt()`, not yet implemented here
- The plan-vs-execution deviation checker
