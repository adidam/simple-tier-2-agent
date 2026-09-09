import os
import json
import re
from openai import OpenAI
from tools import get_stock_info, get_price_history, get_company_profile
from tools import get_web_search
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
}, {
    "type": "function",
    "function": {
        "name": "get_web_search",
        "description": (
            "Search the web for current, qualitative information NOT available from the "
            "other tools — news, management commentary, analyst opinions, competitive "
            "positioning, recent events. Use this for questions about what's happening "
            "with a company recently, or judgment-based topics like moat or management "
            "quality. Do NOT use this for price, valuation ratios, or historical "
            "performance — those come from get_stock_info and get_price_history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'Infosys management changes 2026'"
                }
            },
            "required": ["query"]
        }
    }
}]

PLANNER_SYSTEM_PROMPT = """You are a planning assistant for a stock research agent.
You do NOT have access to any tools yourself — you only decide what should be done.

Available tools the executor can use:
- get_stock_info(ticker): current price and financial ratios (PE, market cap, etc.)
- get_price_history(ticker, period): historical price performance over a time period
- get_company_profile(ticker): business description, sector, industry, employee count
- get_web_search(query): current news, commentary, and qualitative context not
  available from the other tools (e.g. management changes, recent events, analyst
  sentiment, competitive moat discussion)

Given the user's question, write a short numbered plan listing which tool(s) should be
called, in what order, and why each one is needed.

When a question involves comparing or describing MULTIPLE companies, plan the SAME
set of relevant tool calls for EACH company mentioned — do not cover one company more
thoroughly than another just because the question's wording focuses on one of them first.

Only include a get_web_search step when the question genuinely needs current news or
qualitative judgment that the other three tools cannot provide — do not add it reflexively
to every plan.

Example:
Question: "Give me a full picture of AAA.NS: valuation, performance, what its business
does, and how it compares to BBB.NS"

Plan:
[
  {"step": 1, "tool": "get_stock_info", "ticker": "AAA.NS",
      "reason": "Valuation for AAA.NS", "depends_on": []},
  {"step": 2, "tool": "get_price_history", "ticker": "AAA.NS",
      "reason": "Performance for AAA.NS", "depends_on": []},
  {"step": 3, "tool": "get_company_profile", "ticker": "AAA.NS",
      "reason": "What AAA.NS's business does", "depends_on": []},
  {"step": 4, "tool": "get_stock_info", "ticker": "BBB.NS",
      "reason": "Valuation for BBB.NS, to compare against AAA.NS", "depends_on": []},
  {"step": 5, "tool": "get_price_history", "ticker": "BBB.NS",
      "reason": "Performance for BBB.NS, to compare against AAA.NS", "depends_on": []},
  {"step": 6, "tool": "get_company_profile", "ticker": "BBB.NS",
      "reason": "What BBB.NS's business does, for a complete comparison", "depends_on": []}
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
    "You are a financial research assistant. You have access to four tools: "
    "get_stock_info (price and valuation ratios), get_price_history (historical "
    "price performance), get_company_profile (business description, sector, "
    "industry), and get_web_search (current news, commentary, and qualitative "
    "context not available from the other tools). You may call tools multiple "
    "times across turns, reasoning step by step, before deciding on a final answer.\n\n"
    "You will sometimes be given a plan describing an intended sequence of tool "
    "calls before you begin. Treat it as a strong guide, not a rigid script — if "
    "you determine a planned step isn't needed to answer the question, you may "
    "skip it, but explicitly say in your final answer which planned step you "
    "skipped and why.\n\n"
    "If a tool returns an error field, tell the user the ticker wasn't found — "
    "never invent or guess financial data. When discussing qualitative topics "
    "like business moat or competitive position, clearly frame this as your "
    "interpretation based on the available data, not as an established fact. "
    "When you use get_web_search results, briefly note where the information "
    "came from (e.g. 'according to recent coverage') rather than presenting it "
    "with the same certainty as structured financial data."
)

client = wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
))

# MODEL = "anthropic/claude-sonnet-4.6"  # swap to any model OpenRouter hosts, no code change
# MODEL = "meta/muse-spark-1.2"
# MODEL = "moonshotai/kimi-k3"  # swap to any model OpenRouter hosts, no code change
PLANNER_MODEL = "qwen/qwen3-235b-a22b-2507"   # cheap, paid, reliable
EXECUTOR_MODEL = "qwen/qwen3-235b-a22b-2507"  # the reasoning-heavy loop

# low-stakes only
CHEAP_MODEL = "openrouter/free"


def get_plan(question: str) -> str:
    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        max_tokens=2048,   # planning output should be short — this is generous
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
        return
    covered = set()
    for tool, ticker, substituted_from in executed_calls:
        covered.add((tool, ticker))
        if substituted_from:
            covered.add((tool, substituted_from))

    for step in planned_steps:
        tool, ticker = step.get("tool"), step.get("ticker")
        if ticker is None:
            continue
        if (tool, ticker) not in covered:
            print(
                f"⚠️  Plan deviation: step {step['step']} ({tool} on {ticker}) was planned but never executed.")


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "get_stock_info":
        return get_stock_info(args["ticker"])
    elif name == "get_price_history":
        return get_price_history(args["ticker"], args.get("period", "6mo"))
    elif name == "get_company_profile":
        return get_company_profile(args["ticker"])
    elif name == "get_web_search":
        return get_web_search(args["query"])
    else:
        return {"error": f"Unknown tool: {name}"}


def is_not_found_error(result: dict) -> bool:
    err = (result.get("error") or "").lower()
    return "not found" in err or "no profile data" in err or "unable to fetch" in err


def suggest_ticker_correction(bad_ticker: str) -> str | None:

    try:
        response = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a ticker-symbol correction assistant. Given a stock ticker that "
                    "returned no data, suggest the most likely correct NSE ticker symbol "
                    "(ending in .NS). Respond with ONLY the ticker, or the single word UNKNOWN "
                    "if you have no confident guess. No explanation."
                )},
                {"role": "user", "content": f"This ticker returned no data: {bad_ticker}"}
            ])
        suggestion = response.choices[0].message.content.strip()
        return None if suggestion.upper() == "UNKNOWN" else suggestion
    except Exception as e:
        print(
            f"⚠️ Correction suggestion unavailable ({e}); proceeding without a suggested fix.")
        return None


@traceable
def ask(question: str):
    plan_text = get_plan(question)
    planned_steps = parse_plan(plan_text)
    print(f"Plan for question '{question}':\n{plan_text}\n")

    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"},
    ]

    max_iterations = 5
    executed_calls = []
    ticker_corrections = {}  # bad_ticker -> accepted correction, remembered for this run

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=EXECUTOR_MODEL, tools=tools, messages=messages)
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        print(f"Model requested {len(msg.tool_calls)} tool call(s): "
              f"{[tc.function.name for tc in msg.tool_calls]}")

        for tool_call in msg.tool_calls:
            substituted_from = None

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": "Model returned malformed tool arguments"})
                })
                continue

            original_ticker = args.get("ticker")

            # already corrected once this run? apply silently, no re-prompt
            if original_ticker in ticker_corrections:
                substituted_from = original_ticker
                args["ticker"] = ticker_corrections[original_ticker]
                print(
                    f"↪️  Reusing accepted correction: {original_ticker} → {args['ticker']}")

            try:
                result = dispatch_tool(tool_call.function.name, args)
            except Exception as e:
                result = {"error": f"Tool execution failed: {e}"}

            if original_ticker and original_ticker not in ticker_corrections and is_not_found_error(result):
                suggestion = suggest_ticker_correction(original_ticker)
                if suggestion:
                    answer = input(
                        f"\n⚠️  '{original_ticker}' returned no data. Did you mean "
                        f"'{suggestion}'? [y] accept and retry / [n] stop: "
                    ).strip().lower()

                    if answer == "y":
                        ticker_corrections[original_ticker] = suggestion
                        args["ticker"] = suggestion
                        substituted_from = original_ticker
                        try:
                            result = dispatch_tool(
                                tool_call.function.name, args)
                            print(
                                f"✅ Using {suggestion} instead of {original_ticker}")
                        except Exception as e:
                            result = {
                                "error": f"Tool execution failed after substitution: {e}"}
                    else:
                        executed_calls.append(
                            (tool_call.function.name, args.get("ticker"), substituted_from))
                        report_plan_deviation(planned_steps, executed_calls)
                        return f"Stopped: '{original_ticker}' wasn't found and the suggested correction was declined."

            executed_calls.append(
                (tool_call.function.name, args.get("ticker"), substituted_from))
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    report_plan_deviation(planned_steps, executed_calls)
    return "I wasn't able to resolve this within the allowed number of steps."


if __name__ == "__main__":
    import sys
    question = sys.argv[1]
    plan = ask(question)
    print(plan)
