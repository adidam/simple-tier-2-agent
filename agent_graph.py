import os
import sys

from agent import PLANNER_SYSTEM_PROMPT, is_not_found_error, suggest_ticker_correction
from agent import PLANNER_MODEL, EXECUTOR_MODEL, client
from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_stock_info, get_price_history, get_company_profile, get_web_search

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()


# ── 1. STATE ──────────────────────────────────────────────
# Replaces the `messages = [...]` list from agent.py. add_messages is a
# reducer: when a node returns {"messages": [...]}, LangGraph APPENDS those
# to the existing list, same effect as your messages.append(...) calls.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── 2. TOOLS ──────────────────────────────────────────────
# @tool reads each function's docstring + type hints and builds the JSON
# schema automatically — this replaces the hand-written schema dicts you
# maintained by hand in agent.py's `tools` list.

@tool
def stock_info(ticker: str) -> dict:
    """Get current price and key financial ratios (PE, market cap, etc.) for a stock ticker (NSE tickers end in .NS)."""
    return handle_ticker_error(get_stock_info, ticker, "stock_info")


@tool
def price_history(
    ticker: str,
    period: Literal["1d", "5d", "1mo", "3mo", "6mo",
                    "1y", "2y", "5y", "10y", "ytd", "max"] = "6mo"
) -> dict:
    """Fetch historical price performance for a stock ticker over a specified period."""
    return handle_ticker_error(lambda t: get_price_history(t, period), ticker, "price_history")


@tool
def company_profile(ticker: str) -> dict:
    """Get a company's business description, sector, industry, and employee count. Use for 'what does this business do' questions, NOT price or valuation."""
    return handle_ticker_error(get_company_profile, ticker, "company_profile")


@tool
def web_search(query: str) -> dict:
    """Search the web for current news, commentary, or qualitative context not available from the other tools (e.g. management changes, recent events, analyst sentiment)."""
    return get_web_search(query)


tools = [stock_info, price_history, company_profile, web_search]

# ── 3. MODEL ──────────────────────────────────────────────
# Same OpenRouter setup as agent.py, just via LangChain's ChatOpenAI wrapper
# instead of the raw OpenAI SDK client.
# imported from agent.py

model = ChatOpenAI(
    model=EXECUTOR_MODEL,
    max_tokens=4096,   # a bit more room for synthesis than the planner needs
    base_url="https://openrouter.ai/api/v1",
    api_key=(lambda: os.environ["OPENROUTER_API_KEY"]),
).bind_tools(tools)

EXECUTOR_SYSTEM_PROMPT = (
    "You are a financial research assistant. You have access to four tools: "
    "stock_info (price and valuation ratios), price_history (historical price "
    "performance), company_profile (business description, sector, industry), "
    "and web_search (current news, commentary, and qualitative context not "
    "available from the other tools). You may call tools multiple times across "
    "turns, reasoning step by step, before deciding on a final answer.\n\n"
    "If a tool returns an error field, tell the user the ticker wasn't found — "
    "never invent or guess financial data. When discussing qualitative topics "
    "like business moat or competitive position, clearly frame this as your "
    "interpretation based on the available data, not as an established fact. "
    "When you use web_search results, briefly note where the information came "
    "from (e.g. 'according to recent coverage') rather than presenting it with "
    "the same certainty as structured financial data."
)

# module-level, reset per process — good enough for a CLI run
_ticker_corrections: dict[str, str] = {}


def handle_ticker_error(get_data_fn, ticker: str, tool_name: str) -> dict:
    # if this ticker was already corrected earlier in this run, reuse it silently
    if ticker in _ticker_corrections:
        return get_data_fn(_ticker_corrections[ticker])

    result = get_data_fn(ticker)
    if not is_not_found_error(result):
        return result

    suggestion = suggest_ticker_correction(ticker)
    answer = interrupt({
        "type": "ticker_correction",
        "tool": tool_name,
        "original_ticker": ticker,
        "suggested_ticker": suggestion,
    }).strip()

    if answer.lower() == "y" and suggestion:
        _ticker_corrections[ticker] = suggestion
        return get_data_fn(suggestion)
    elif answer.lower() == "n" or not answer:
        return {"error": f"Stopped: '{ticker}' not found, correction declined."}
    else:
        _ticker_corrections[ticker] = answer.upper()
        return get_data_fn(answer.upper())

# ── 4. NODES ──────────────────────────────────────────────
# This is the model-calling half of your old `for` loop, as its own function.


def get_plan(question: str) -> str:
    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        max_tokens=2048,         # planning output should be short — this is generous
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        # no `tools` — forces a text/JSON plan, not a tool_calls response, same as agent.py
    )
    return response.choices[0].message.content


def plan_node(state: AgentState) -> AgentState:
    question = state["messages"][-1].content
    plan_text = get_plan(question)
    print(f"Plan for question '{question}':\n{plan_text}\n")
    return {"messages": [{"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"}]}


def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    # prepend the system prompt only if it's not already there (first call)
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT)] + messages

    response = model.invoke(messages)
    return {"messages": [response]}


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
        result = graph.invoke({"messages": [{"role": "user", "content": question}]},
                              config=config)

        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            suggestion_text = f"Did you mean '{payload['suggested_ticker']}'? " if payload[
                'suggested_ticker'] else "No suggestion available. "
            answer = input(
                f"\n⚠️  '{payload['original_ticker']}' returned no data. {suggestion_text}"
                f"[y] accept / [n] stop / or type the correct ticker: "
            ).strip()
            result = graph.invoke(Command(resume=answer), config=config)
    except GraphRecursionError:
        return "I wasn't able to resolve this within the allowed number of steps."

    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask(sys.argv[1]))
