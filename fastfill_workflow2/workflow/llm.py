"""One LLM request through the official OpenAI Responses API, with retries and a usage record.

Settings come from ``credentials.official_api`` (its own file, never the process environment); a call whose
base URL is not api.openai.com is refused. A rejected key (401/403), an exhausted quota or a missing setting raises
``credentials.FatalAPIError``, which the planner and drafter do not catch, so a run stops instead of falling back
to rules. Transient failures (429 rate limit, 5xx, timeouts, an incomplete answer) are retried with backoff.
Usage is reported in chat-completions field names, the format the cost ledgers read, plus ``cost_usd``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .credentials import FatalAPIError, cost_usd, official_api

USER_AGENT = "fastfill-workflow/1.0"


def _usage(raw: dict | None) -> dict:
    raw = raw or {}
    return {"prompt_tokens": raw.get("input_tokens"), "completion_tokens": raw.get("output_tokens"),
            "total_tokens": raw.get("total_tokens"),
            "prompt_tokens_details": {"cached_tokens": (raw.get("input_tokens_details") or {}).get("cached_tokens")},
            "completion_tokens_details": {"reasoning_tokens": (raw.get("output_tokens_details") or {}).get("reasoning_tokens")}}


def _text(answer: dict) -> str:
    return "".join(part.get("text", "") for item in answer.get("output") or [] if item.get("type") == "message"
                   for part in item.get("content") or [] if part.get("type") == "output_text")


def ask(instructions: str, prompt: str, *, model: str | None = None, base_url: str | None = None,
        api_key: str | None = None, reasoning_effort: str | None = None,
        attempts: int = 6, timeout_s: float = 600.) -> tuple[str, dict]:
    """Returns (text, call) with call = {model, usage, cost_usd, latency_s, failed_attempts}."""
    if not (model and base_url and api_key):
        settings = official_api(model=model)
        model, base_url, api_key = model or settings["model"], base_url or settings["base_url"], \
            api_key or settings["api_key"]
    if urlparse(base_url).hostname != "api.openai.com":
        raise FatalAPIError("the workflow runs on the official OpenAI API only")
    body = {"model": model, "instructions": instructions, "store": False,
            "input": [{"role": "user", "content": prompt}],
            "reasoning": {"effort": reasoning_effort or os.environ.get("WORKFLOW_REASONING_EFFORT", "medium")}}
    data = json.dumps(body).encode("utf-8")
    url = base_url.rstrip("/") + "/responses"
    start, problem = time.perf_counter(), None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data, {"Authorization": f"Bearer {api_key}",
                                                         "Content-Type": "application/json", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                answer = json.load(response)
            text = _text(answer)
            usage = _usage(answer.get("usage"))
            if answer.get("status") == "completed" and text:
                return text, {"model": model, "usage": usage, "cost_usd": cost_usd(usage, model),
                              "latency_s": round(time.perf_counter() - start, 3), "failed_attempts": attempt}
            problem = answer.get("error") or answer.get("incomplete_details") or "empty response"
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300]
            problem = f"HTTP {exc.code}: {detail!r}"
            if exc.code in (401, 403, 404) or b"insufficient_quota" in detail:  # key, access, unknown model, quota
                raise FatalAPIError(f"OpenAI rejected the request: HTTP {exc.code} {detail[:160]!r}") from None
            if exc.code == 400:  # this request is malformed (e.g. too long): retrying will not help
                break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(min(60., 5. * 2 ** attempt))
    raise RuntimeError(f"LLM request failed: {str(problem)[:300]}")
