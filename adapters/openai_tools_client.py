"""OpenAI tool-calling bridge for the valuation dashboard.

This module lets the dashboard optionally route analysis through OpenAI.
The model will request tool calls; we execute them locally (valuation engine),
then send tool outputs back until we get the final JSON payload.

Security:
- Never log or print the API key.
- The dashboard stores the key only in memory (Streamlit session_state) unless
  the user sets OPENAI_API_KEY in their environment.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIError(RuntimeError):
    pass


class OpenAIRateLimit(OpenAIError):
    pass


class OpenAITokenLimit(OpenAIError):
    pass


class OpenAIToolsUnavailable(OpenAIError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str


def _post_json(url: str, api_key: str, payload: dict, timeout_s: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        _raise_openai_http_error(e.code, body)
        raise
    except urllib.error.URLError as e:
        raise OpenAIError("Unable to fetch data. Try again in a moment.") from e


def _raise_openai_http_error(status: int, body: str) -> None:
    msg = body.lower()
    if status == 401 or "invalid_api_key" in msg or "unauthorized" in msg:
        raise OpenAIError("OpenAI API key invalid or missing.")
    if status == 429 or "rate limit" in msg:
        raise OpenAIRateLimit("Rate limit reached. Please wait a moment before analyzing another stock.")
    if status == 400 and ("maximum context length" in msg or "context_length_exceeded" in msg):
        raise OpenAITokenLimit("This analysis used too many tokens. Try analyzing fewer stocks at once.")
    if "unknown tool" in msg or "tool" in msg and "not found" in msg:
        raise OpenAIToolsUnavailable("OpenAI MCP tools not available. Check API documentation.")
    raise OpenAIError("Unable to fetch data. Try again in a moment.")


def get_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.strip()
    env = os.getenv("OPENAI_API_KEY")
    return env.strip() if env else None


def tool_specs() -> list[dict]:
    """OpenAI Responses API tool specs."""
    # Names match the user's requested tool list, but execution is local.
    return [
        {
            "type": "function",
            "name": "valuationengine_fetch_fundamentals",
            "description": "Fetch current fundamentals for a public company by ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
        {
            "type": "function",
            "name": "valuationengine_run_dcf",
            "description": "Run a DCF valuation for a ticker with optional overrides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "revenue_growth": {"type": ["number", "null"]},
                    "operating_margin": {"type": ["number", "null"]},
                    "terminal_growth": {"type": "number"},
                    "risk_free_rate": {"type": "number"},
                    "equity_risk_premium": {"type": "number"},
                    "use_exit_multiple": {"type": "boolean"},
                    "exit_ev_ebitda_multiple": {"type": "number"},
                    "projection_years": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
        {
            "type": "function",
            "name": "valuationengine_run_scenario",
            "description": "Run bull/base/bear DCF scenarios for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "growth_delta": {"type": "number"},
                    "margin_delta": {"type": "number"},
                },
                "required": ["ticker"],
            },
        },
        {
            "type": "function",
            "name": "valuationengine_run_reverse_dcf",
            "description": "Run reverse DCF for a ticker and field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "field": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
        {
            "type": "function",
            "name": "valuationengine_run_lbo",
            "description": "Run LBO for a ticker with optional overrides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "entry_ev_ebitda_multiple": {"type": "number"},
                    "exit_ev_ebitda_multiple": {"type": "number"},
                    "debt_pct_purchase": {"type": "number"},
                    "interest_rate": {"type": "number"},
                    "revenue_growth": {"type": ["number", "null"]},
                    "operating_margin": {"type": ["number", "null"]},
                    "hold_period_years": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    ]


def _extract_tool_calls(response: dict) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in response.get("output", []):
        if item.get("type") == "function_call":
            name = item.get("name")
            args_str = item.get("arguments") or "{}"
            call_id = item.get("call_id") or item.get("id") or ""
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except json.JSONDecodeError:
                args = {}
            if name and call_id:
                calls.append(ToolCall(name=name, arguments=args, call_id=call_id))
    return calls


def _extract_final_json(response: dict) -> dict | None:
    # We ask the model to return JSON only; it should appear in output_text.
    text = response.get("output_text")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run_tool_calling_session(
    api_key: str,
    user_prompt: str,
    tool_executor,
    model_preference: list[str] | None = None,
    max_steps: int = 6,
) -> dict:
    """
    Run OpenAI tool-calling loop.

    tool_executor: (tool_name: str, arguments: dict) -> dict
    """
    models = model_preference or ["gpt-4o", "gpt-4-turbo"]

    system = (
        "You are a financial analysis assistant. When asked to analyze a stock, "
        "use the valuation engine tools to fetch data and run analyses. "
        "Return only the structured JSON data, no narrative or explanation."
    )

    last_error: Exception | None = None
    for model in models:
        try:
            response_id: str | None = None
            inputs = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
            for _step in range(max_steps):
                payload = {
                    "model": model,
                    "input": inputs,
                    "tools": tool_specs(),
                }
                if response_id:
                    payload["previous_response_id"] = response_id
                resp = _post_json(OPENAI_RESPONSES_URL, api_key, payload)
                response_id = resp.get("id") or response_id

                final = _extract_final_json(resp)
                if final is not None:
                    return final

                calls = _extract_tool_calls(resp)
                if not calls:
                    raise OpenAIError("Unable to fetch data. Try again in a moment.")

                # Add tool call outputs as the next input items.
                tool_outputs = []
                for call in calls:
                    result = tool_executor(call.name, call.arguments)
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result),
                        }
                    )
                inputs = tool_outputs

            raise OpenAIError("Unable to fetch data. Try again in a moment.")
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise OpenAIError("Unable to fetch data. Try again in a moment.")

