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
called, in what order, and why each one is needed. If a step depends on the result of an
earlier step (e.g. "compare X and Y, then look up the winner's history"), say so explicitly.
Do not attempt to answer the question yourself — only produce the plan.

Output the plan as a JSON array. Each step has:
- "step": integer id
- "tool": the tool name to call
- "ticker": the ticker (if you know it yet)
- "reason": why this step is needed
- "depends_on": a list of step ids whose OUTPUT is required before this step's
  arguments can be determined, or [] if none.

A step depends on another ONLY if you literally cannot fill in its arguments
without seeing the earlier step's result. Comparing results later does not count.

Respond with ONLY the JSON array, no other text."""

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
        {
            "role": "system",
            "content": (
                "You are a financial data assistant. Use the get_stock_info, get_price_history tools "
                "you can call the tools multiple times across turns if you need to reason step by step, "
                "before before deciding on the final tool call "
                "to answer questions about stock prices and ratios. If the tool "
                "returns an error field, tell the user the ticker wasn't found — "
                "never invent or guess financial data."
            )
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"}
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
