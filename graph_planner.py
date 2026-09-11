"""
graph_planner.py — planning call, plan parsing, and plan-vs-execution
deviation checking. Self-contained copies of what agent.py (now legacy)
originally validated, adapted for graph state instead of a plain list.
"""

import os
import json
import re
from openai import OpenAI

from graph_prompts import PLANNER_SYSTEM_PROMPT
from graph_tools import _ticker_corrections

from dotenv import load_dotenv

load_dotenv()

# paid, reliable, pinned deliberately —
PLANNER_MODEL = "qwen/qwen3-235b-a22b-2507"
# openrouter/auto and openrouter/free can silently serve a DIFFERENT underlying
# model on every call, even within one conversation. Planning and synthesis are
# the highest-stakes steps in this pipeline, so they stay on a fixed model.

# reads OPENROUTER_API_KEY from env
_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                 api_key=(lambda: os.environ["OPENROUTER_API_KEY"]))


def get_plan(question: str) -> str:
    """One call, no tools attached — forces a text/JSON plan rather than a
    tool_calls response."""
    response = _client.chat.completions.create(
        model=PLANNER_MODEL,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def parse_plan(plan_text: str) -> list[dict] | None:
    """Strip the ```json fence the model wraps the plan in, parse to a list
    of step dicts. Returns None if parsing fails."""
    match = re.search(r"\[.*\]", plan_text, re.DOTALL)
    json_str = match.group(0) if match else plan_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def extract_executed_calls(messages) -> list[tuple[str, str | None, str | None]]:
    """Reconstruct (tool_name, ticker_used, substituted_from) for every tool
    call actually made, by reading it back out of the final message history."""
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


def report_plan_deviation(planned_steps: list[dict] | None, executed_calls: list[tuple]) -> None:
    """Print a warning for any planned step (with a known ticker) that never
    executed — an approved ticker substitution counts as fulfilling the step."""
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


def plan_node(state: dict) -> dict:
    """Graph node: generate a plan, print it, save into state under both
    'messages' (context for the executor) and 'plan' (for deviation
    checking after execution finishes)."""
    question = state["messages"][-1].content
    plan_text = get_plan(question)
    print(f"Plan for question '{question}':\n{plan_text}\n")
    return {
        "messages": [{"role": "assistant", "content": f"Here is my plan before executing:\n{plan_text}"}],
        "plan": plan_text,
    }
