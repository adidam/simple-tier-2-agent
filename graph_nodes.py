import os
from agent import EXECUTOR_MODEL
from graph_tools import get_stock_info, get_price_history, get_company_profile, get_web_search
from graph_prompts import EXECUTOR_SYSTEM_PROMPT
from graph_tools import _ticker_corrections
from langchain_core.messages import SystemMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# ── 1. STATE ──────────────────────────────────────────────
# Replaces the `messages = [...]` list from agent.py. add_messages is a
# reducer: when a node returns {"messages": [...]}, LangGraph APPENDS those
# to the existing list, same effect as your messages.append(...) calls.


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str


_tools = [get_stock_info, get_price_history,
          get_company_profile, get_web_search]


# ── 3. MODEL ──────────────────────────────────────────────
# Same OpenRouter setup as agent.py, just via LangChain's ChatOpenAI wrapper
# instead of the raw OpenAI SDK client.
# imported from agent.py

model = ChatOpenAI(
    model=EXECUTOR_MODEL,
    max_tokens=4096,   # a bit more room for synthesis than the planner needs
    base_url="https://openrouter.ai/api/v1",
    api_key=(lambda: os.environ["OPENROUTER_API_KEY"]),
).bind_tools(_tools)


# ── 4. NODES ──────────────────────────────────────────────
# This is the model-calling half of your old `for` loop, as its own function.

def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    # prepend the system prompt only if it's not already there (first call)
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT)] + messages

    response = model.invoke(messages)
    return {"messages": [response]}


def extract_executed_calls(messages):
    calls = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc["name"]
            ticker = tc["args"].get("ticker")
            substituted_from = None
            if ticker in _ticker_corrections:
                substituted_from = ticker
                ticker = _ticker_corrections[ticker]
            calls.append((name, ticker, substituted_from))
    return calls
