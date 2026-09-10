
import sys

from agent import parse_plan, report_plan_deviation
from graph_nodes import plan_node, call_model, extract_executed_calls, AgentState, tools
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


from langgraph.errors import GraphRecursionError


from dotenv import load_dotenv
load_dotenv()


# ToolNode is prebuilt — it replaces your hand-written dispatch_tool if/elif
# chain AND already handles multiple simultaneous tool calls in one turn.
tool_node = ToolNode(tools)

# ── 5. WIRE THE GRAPH ─────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "plan")             # was: START → call_model
builder.add_edge("plan", "call_model")      # new fixed edge
# tools_condition is prebuilt — same job as your `if not msg.tool_calls: return`
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")  # the cycle replaces your for loop

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ── 6. RUN IT ─────────────────────────────────────────────


def ask(question: str) -> str:
    config = {"configurable": {
        "thread_id": "cli-session"}, "recursion_limit": 15}
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            suggestion_text = f"Did you mean '{payload['suggested_ticker']}'? " if payload[
                'suggested_ticker'] else "No suggestion available. "
            answer = input(
                f"\n⚠️  '{payload['original_ticker']}' returned no data. {suggestion_text}"
                f"[y] accept / [n] stop / or type the correct ticker: "
            ).strip()
            result = graph.invoke(Command(resume=answer), config=config)

        planned_steps = parse_plan(result.get("plan", ""))
        executed_calls = extract_executed_calls(result["messages"])
        report_plan_deviation(planned_steps, executed_calls)

        return result["messages"][-1].content
    except GraphRecursionError:
        return "I wasn't able to resolve this within the allowed number of steps."

    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask(sys.argv[1]))
