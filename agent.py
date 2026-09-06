import os
import json
import re
from openai import OpenAI
from tools import get_stock_info, get_price_history, get_company_profile
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith import traceable

load_dotenv()

tools = [{
    "type": "function",
    "function": {
        "name": "get_stock_info",
        "description": "Get current price and key financial ratios for a stock ticker (NSE tickers end in .NS).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. INFY.NS"
                }
            },
            "required": ["ticker"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "get_price_history",
        "description": "Fetch price history for a stock ticker over a specified period.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. INFY.NS"
                },
                "period": {
                    "type": "string",
                    "description": "Time period for the price history (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y')"
                }
            },
            "required": ["ticker", "period"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "get_company_profile",
        "description": (
            "Get a company's business description, sector, industry, and size "
            "(employee count). Use this for questions about what a company does, "
            "its industry, or qualitative business characteristics like a competitive "
            "moat — NOT for price, valuation ratios, or financial performance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. INFY.NS"
                }
            },
            "required": ["ticker"]
        }
    }
}]

PLANNER_SYSTEM_PROMPT = """You are a planning assistant for a stock research agent.
You do NOT have access to any tools yourself — you only decide what should be done.

Available tools the executor can use:
- get_stock_info(ticker): current price and financial ratios (PE, market cap, etc.)
- get_price_history(ticker, period): historical price performance over a time period
- get_company_profile(ticker): business description, sector, industry, employee count

Given the user's question, write a short numbered plan listing which tool(s) should be
called, in what order, and why each one is needed.

When a question involves comparing or describing MULTIPLE companies, plan the SAME
set of relevant tool calls for EACH company mentioned — do not cover one company more
thoroughly than another just because the question's wording focuses on one of them first.

Example:
Question: "Give me a full picture of AAA.NS: valuation, performance, what its business
does, and how it compares to BBB.NS"

Plan:
[
  {"step": 1, "tool": "get_stock_info", "ticker": "AAA.NS", "reason": "Valuation for AAA.NS", "depends_on": []},
  {"step": 2, "tool": "get_price_history", "ticker": "AAA.NS", "reason": "Performance for AAA.NS", "depends_on": []},
  {"step": 3, "tool": "get_company_profile", "ticker": "AAA.NS", "reason": "What AAA.NS's business does", "depends_on": []},
  {"step": 4, "tool": "get_stock_info", "ticker": "BBB.NS", "reason": "Valuation for BBB.NS, to compare against AAA.NS", "depends_on": []},
  {"step": 5, "tool": "get_price_history", "ticker": "BBB.NS", "reason": "Performance for BBB.NS, to compare against AAA.NS", "depends_on": []},
  {"step": 6, "tool": "get_company_profile", "ticker": "BBB.NS", "reason": "What BBB.NS's business does, for a complete comparison", "depends_on": []}
]

Notice BBB.NS gets a company profile step too, even though the question only explicitly
asked "what its business does" about AAA.NS — a genuine comparison requires understanding
both businesses, not just one.

Output the plan as a JSON array. Each step has:
- "step": integer id
- "tool": the tool name to call
- "ticker": the ticker (if you know it yet)
- "reason": why this step is needed
- "depends_on": a list of step ids whose OUTPUT is required before this step's
  arguments can be determined, or [] if none.

A step depends on another ONLY if you literally cannot fill in its arguments
without seeing the earlier step's result. Comparing results later does not count.

Respond with ONLY the JSON array, no other text — the example above is for your
reference only, do not include it in your response."""

EXECUTOR_SYSTEM_PROMPT = (
    "You are a financial research assistant. You have access to three tools: "
    "get_stock_info (price and valuation ratios), get_price_history (historical "
    "price performance), and get_company_profile (business description, sector, "
    "industry). You may call tools multiple times across turns, reasoning step by "
    "step, before deciding on a final answer.\n\n"
    "You will sometimes be given a plan describing an intended sequence of tool "
    "calls before you begin. Treat it as a strong guide, not a rigid script — if "
    "you determine a planned step isn't needed to answer the question, you may "
    "skip it, but explicitly say in your final answer which planned step you "
    "skipped and why.\n\n"
    "If a tool returns an error field, tell the user the ticker wasn't found — "
    "never invent or guess financial data. When discussing qualitative topics "
    "like business moat or competitive position, clearly frame this as your "
    "interpretation based on the available data, not as an established fact."
)

client = wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
))

# MODEL = "anthropic/claude-sonnet-4.6"  # swap to any model OpenRouter hosts, no code change
# MODEL = "meta/muse-spark-1.2"
MODEL = "moonshotai/kimi-k3"  # swap to any model OpenRouter hosts, no code change


def get_plan(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        # deliberately no `tools=` here — forces a text plan, not a tool_calls response
    )
    return response.choices[0].message.content


def parse_plan(plan_text: str):
    match = re.search(r"\[.*\]", plan_text, re.DOTALL)
    json_str = match.group(0) if match else plan_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None  # if parsing fails, we just skip the comparison later


def report_plan_deviation(planned_steps, executed_calls):
    if not planned_steps:
        return  # plan failed to parse — nothing to compare against
    executed_set = set(executed_calls)
    for step in planned_steps:
        tool, ticker = step.get("tool"), step.get("ticker")
        if ticker is None:
            continue  # dependent step — ticker was only knowable at runtime, can't compare directly
        if (tool, ticker) not in executed_set:
            print(
                f"⚠️  Plan deviation: step {step['step']} ({tool} on {ticker}) was planned but never executed.")


@traceable
def ask(question: str):
    plan_text = get_plan(question)   # your new planning call
    planned_steps = parse_plan(plan_text)
    print(f"Plan for question '{question}':\n{plan_text}\n")
    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"},
    ]

    max_iterations = 5
    executed_calls = []  # (tool_name, ticker) actually run

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            tools=tools,
            messages=messages
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # model decided it has enough info — this is the exit condition
            return msg.content

        print(
            f"Model requested {len(msg.tool_calls)} tool call(s): {[tc.function.name for tc in msg.tool_calls]}")

        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = {"error": "Model returned malformed tool arguments"}
            else:
                executed_calls.append(
                    (tool_call.function.name, args.get("ticker")))
                try:
                    if tool_call.function.name == "get_stock_info":
                        result = get_stock_info(args["ticker"])
                    elif tool_call.function.name == "get_price_history":
                        result = get_price_history(
                            args["ticker"], args.get("period", "6mo"))
                    elif tool_call.function.name == "get_company_profile":
                        result = get_company_profile(args["ticker"])
                    else:
                        result = {
                            "error": f"Unknown tool: {tool_call.function.name}"}
                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    report_plan_deviation(planned_steps, executed_calls)
    # if we exit the for-loop without returning, the cap was hit
    return "I wasn't able to resolve this within the allowed number of steps."


if __name__ == "__main__":
    import sys
    question = sys.argv[1]
    plan = ask(question)
    print(plan)
