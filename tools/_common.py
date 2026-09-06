"""Small utility to fetch stock price and key ratios.

Uses Yahoo Finance unofficial JSON endpoints with a fallback to yfinance if available.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

try:
    import requests
except Exception:  # pragma: no cover - best-effort import
    requests = None  # type: ignore

import urllib.request
import urllib.error


def _fetch_url(url: str, timeout: int = 10, headers: Optional[Dict[str, str]] = None):
    """Fetch a URL returning (status_code, text, json_or_None).

    Uses `requests` if available, otherwise `urllib` from the stdlib.
    """
    headers = headers or {}
    if requests is not None:
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            text = resp.text
            status = getattr(resp, "status_code", None)
            try:
                j = resp.json()
            except Exception:
                j = None
            return status, text, j
        except Exception:
            return None, "", None
    # urllib fallback
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            text = raw.decode("utf-8", errors="replace")
            status = getattr(r, "status", None)
            try:
                j = json.loads(text)
            except Exception:
                j = None
            return status, text, j
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            text = str(e)
        return getattr(e, "code", None), text, None
    except Exception:
        return None, "", None


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _print_json(d: Dict[str, Any]) -> None:
    print(json.dumps(d, indent=2, sort_keys=True, default=str))
