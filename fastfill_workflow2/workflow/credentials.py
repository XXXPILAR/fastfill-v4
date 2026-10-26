"""Official OpenAI API settings for the workflow, read from their own file (the key is never printed).

The file is ``Fastfill workflow2/.env`` if present, else ``Fastfill workflow/.env`` (else ``~/.fastfill_openai.env``) with ``KEY=VALUE`` lines::

    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-5.6-sol          # optional, this is the default
    OPENAI_BASE_URL=https://api.openai.com/v1   # optional; any other host is refused

It is separate from ``~/.fastfill_api.env`` (the baselines' relay), and the process environment is ignored on
purpose, so a relay key exported for the baselines can never reach the workflow.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

OFFICIAL_BASE_URL = "https://api.openai.com/v1"
# workflow2 has no key of its own: its own .env if one is ever put there, else the original workflow's (read in place,
# never copied)
_OWN_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"                          # Fastfill workflow2/.env
_V1_ENV_FILE = Path(__file__).resolve().parents[2] / "Fastfill workflow" / ".env"      # Fastfill workflow/.env
DEFAULT_ENV_FILE = _OWN_ENV_FILE if _OWN_ENV_FILE.is_file() or not _V1_ENV_FILE.is_file() else _V1_ENV_FILE
HOME_ENV_FILE = Path.home() / ".fastfill_openai.env"
DEFAULT_MODEL = "gpt-5.6-sol"
# USD per 1M tokens (input, cached input, output; reasoning is billed as output), from the model page
# https://developers.openai.com/api/docs/models/gpt-5.6-sol (promotional pricing, at least to 2026-11-21)
PRICES_USD = {"gpt-5.6-sol": (4.0, 0.4, 20.0)}


class FatalAPIError(Exception):
    """Settings or account problems (no key, non-OpenAI host, rejected key, exhausted quota). Not a subclass of
    the errors the planner and drafter catch, so it stops a run instead of silently falling back to rules."""


def _read(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.split(" #", 1)[0].strip().strip("'\"")
    return values


def official_api(path: str | Path | None = None, model: str | None = None) -> dict[str, str]:
    """{"model", "base_url", "api_key"} for ``llm.ask``; raises if the key is missing or the host is not OpenAI."""
    if path:
        path = Path(path).expanduser()
    else:
        path = DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.is_file() or not HOME_ENV_FILE.is_file() else HOME_ENV_FILE
    if not path.is_file():
        raise FatalAPIError(f"no OpenAI settings file at {path} (put OPENAI_API_KEY=... in it)")
    values = _read(path)
    if not values.get("OPENAI_API_KEY"):
        raise FatalAPIError(f"{path} has no OPENAI_API_KEY")
    base_url = values.get("OPENAI_BASE_URL", OFFICIAL_BASE_URL).rstrip("/")
    if urlparse(base_url).hostname != "api.openai.com":
        raise FatalAPIError("the workflow runs on the official OpenAI API only (OPENAI_BASE_URL must be api.openai.com)")
    return {"model": model or values.get("OPENAI_MODEL", DEFAULT_MODEL), "base_url": base_url,
            "api_key": values["OPENAI_API_KEY"]}


def cost_usd(usage: dict, model: str) -> float | None:
    """Billed USD of one call from its usage record (chat-completions field names, as ``llm.ask`` reports)."""
    if model not in PRICES_USD or not usage:
        return None
    price_in, price_cached, price_out = PRICES_USD[model]
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    output = usage.get("completion_tokens") or 0
    return ((prompt - cached) * price_in + cached * price_cached + output * price_out) / 1e6
