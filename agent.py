import os
import json
from openai import OpenAI
from tools import fetch_stock_data
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai

load_dotenv()

tools = [{
    "type": "function",
    "function": {
        "name": "fetch_stock_data",
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
}]


client = wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
))

# MODEL = "anthropic/claude-sonnet-4.6"  # swap to any model OpenRouter hosts, no code change
# MODEL = "meta/muse-spark-1.2"
MODEL = "moonshotai/kimi-k3"  # swap to any model OpenRouter hosts, no code change

def ask(question: str):
    messages =  [
      {
        "role": "system",
        "content": (
            "You are a financial data assistant. Use the get_stock_info tool "
            "to answer questions about stock prices and ratios. If the tool "
            "returns an error field, tell the user the ticker wasn't found — "
            "never invent or guess financial data."
        )
      },
      {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        tools=tools,
        messages=messages
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = fetch_stock_data(args["ticker"])

        messages.append(msg)  # the assistant's tool-call turn
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

        final = client.chat.completions.create(
            model=MODEL,
            tools=tools,
            messages=messages
        )
        return final.choices[0].message.content

    return msg.content

if __name__ == "__main__":
    import sys
    print(ask(sys.argv[1]))