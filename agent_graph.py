import os
import sys

from agent import is_not_found_error, suggest_ticker_correction

from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_stock_info, get_price_history, get_company_profile, get_web_search

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI


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
    result = get_stock_info(ticker)
    if is_not_found_error(result):
        suggestion = suggest_ticker_correction(ticker)
        if suggestion:
            answer = interrupt({
                "type": "ticker_correction",
                "tool": "stock_info",
                "original_ticker": ticker,
                "suggested_ticker": suggestion
            })
            if answer == 'y':
                result = get_stock_info(suggestion)
            else:
                result = {
                    "error": f"Stopped: '{ticker}' not found, correction declined."}
    return result


@tool
def price_history(ticker: str, period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "6mo") -> dict:
    """Fetch historical price performance for a stock ticker over a specified period."""
    return get_price_history(ticker, period)


@tool
def company_profile(ticker: str) -> dict:
    """Get a company's business description, sector, industry, and employee count. Use for 'what does this business do' questions, NOT price or valuation."""
    return get_company_profile(ticker)


@tool
def web_search(query: str) -> dict:
    """Search the web for current news, commentary, or qualitative context not available from the other tools (e.g. management changes, recent events, analyst sentiment)."""
    return get_web_search(query)


tools = [stock_info, price_history, company_profile, web_search]

# ── 3. MODEL ──────────────────────────────────────────────
# Same OpenRouter setup as agent.py, just via LangChain's ChatOpenAI wrapper
# instead of the raw OpenAI SDK client.

# MODEL = "openrouter/auto"
MODEL = "openrouter/free"

model = ChatOpenAI(
    model=MODEL,
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

# ── 4. NODES ──────────────────────────────────────────────
# This is the model-calling half of your old `for` loop, as its own function.


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
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
# tools_condition is prebuilt — same job as your `if not msg.tool_calls: return`
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")  # the cycle replaces your for loop

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ── 6. RUN IT ─────────────────────────────────────────────


def ask(question: str) -> str:
    config = {"configurable": {
        "thread_id": "cli-session"}, "recursion_limit": 15}
    result = graph.invoke({"messages": [{"role": "user", "content": question}]},
                          config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = input(
            f"\n⚠️  '{payload['original_ticker']}' returned no data. Did you mean "
            f"'{payload['suggested_ticker']}'? [y] accept and retry / [n] stop: "
        ).strip().lower()
        result = graph.invoke(Command(resume=answer), config=config)

    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask(sys.argv[1]))
