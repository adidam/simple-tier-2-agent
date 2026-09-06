from typing import Dict, Any


def get_web_search(query, max_results=5):
    """
    Perform a web search using the provided query and return the results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns a dict with keys:
      - query: the search query used
      - results: list of {title, url, snippet}
      - error: populated if the search failed or returned nothing
    """
    result: Dict[str, Any] = {"query": query, "results": [], "error": None}

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        result["error"] = f"Web search failed: {e}"
        return result

    if not raw_results:
        result["error"] = f"No search results found for: {query}"
        return result

    trimmed = []
    for r in raw_results:
        trimmed.append({
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")
        })

    result["results"] = trimmed
    return result
