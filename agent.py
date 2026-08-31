import os
import json
from openai import OpenAI
from tools import get_stock_info, get_price_history
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
}]


client = wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
))

# MODEL = "anthropic/claude-sonnet-4.6"  # swap to any model OpenRouter hosts, no code change
# MODEL = "meta/muse-spark-1.2"
MODEL = "moonshotai/kimi-k3"  # swap to any model OpenRouter hosts, no code change

@traceable
def ask(question: str):
    messages =  [
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
      {"role": "user", "content": question}
    ]
    
    max_iterations = 5   

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

        print(f"Model requested {len(msg.tool_calls)} tool call(s): {[tc.function.name for tc in msg.tool_calls]}")

        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = {"error": "Model returned malformed tool arguments"}
            else:
                try:
                    if tool_call.function.name == "get_stock_info":
                        result = get_stock_info(args["ticker"])
                    elif tool_call.function.name == "get_price_history":
                        result = get_price_history(args["ticker"], args.get("period", "6mo"))
                    else:
                        result = {"error": f"Unknown tool: {tool_call.function.name}"}
                except Exception as e:
                     result = {"error": f"Tool execution failed: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # if we exit the for-loop without returning, the cap was hit
    return "I wasn't able to resolve this within the allowed number of steps."

if __name__ == "__main__":
    import sys
    print(ask(sys.argv[1]))