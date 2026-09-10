import os
from agent import EXECUTOR_MODEL
from graph_tools import get_stock_info, get_price_history, get_company_profile, get_web_search

from graph_tools import _ticker_corrections
from langchain_core.messages import SystemMessage
from agent import PLANNER_SYSTEM_PROMPT, PLANNER_MODEL, client
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


tools = [get_stock_info, get_price_history,
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
).bind_tools(tools)


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
    return {"messages": [{"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"}],
            "plan": plan_text,
            }


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
