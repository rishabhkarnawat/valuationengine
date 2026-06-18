"""OpenAI tool-calling bridge for the valuation dashboard.

The model requests tool calls; we execute them locally (valuation engine),
then send tool outputs back until we get the final JSON payload.

Security:
- Never log or print the API key.
- Keys come from the environment or the dashboard password field only.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


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


def _ssl_context() -> ssl.SSLContext:
    """Use certifi CA bundle when available (fixes macOS python.org SSL errors)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _post_json(url: str, api_key: str, payload: dict, timeout_s: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_context()) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        _raise_openai_http_error(e.code, body)
        raise
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
            raise OpenAIError(
                "SSL certificate verification failed when calling OpenAI. "
                "On macOS, run Applications/Python 3.x/Install Certificates.command."
            ) from e
        raise OpenAIError("Unable to fetch data. Try again in a moment.") from e


def _api_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        if isinstance(data.get("error"), dict):
            return str(data["error"].get("message") or "")
    except json.JSONDecodeError:
        pass
    return ""


def _raise_openai_http_error(status: int, body: str) -> None:
    msg = body.lower()
    detail = _api_error_message(body)

    if status == 401 or "invalid_api_key" in msg or "incorrect api key" in msg:
        raise OpenAIError("OpenAI API key invalid or missing.")
    if status == 429 or "rate limit" in msg:
        raise OpenAIRateLimit("Rate limit reached. Please wait a moment before analyzing another stock.")
    if "maximum context length" in msg or "context_length_exceeded" in msg:
        raise OpenAITokenLimit("This analysis used too many tokens. Try analyzing fewer stocks at once.")
    if "unknown tool" in msg or ("tool" in msg and "not found" in msg):
        raise OpenAIToolsUnavailable("OpenAI MCP tools not available. Check API documentation.")
    if detail:
        raise OpenAIError(detail)
    raise OpenAIError("Unable to fetch data. Try again in a moment.")


def get_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        key = explicit.strip()
        return key or None
    env = os.getenv("OPENAI_API_KEY")
    return env.strip() if env else None


def tool_specs() -> list[dict]:
    """OpenAI Chat Completions function tool definitions."""
    def fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or [],
                },
            },
        }

    return [
        fn(
            "valuationengine_fetch_fundamentals",
            "Fetch current fundamentals for a public company by ticker.",
            {"ticker": {"type": "string"}},
            ["ticker"],
        ),
        fn(
            "valuationengine_run_dcf",
            "Run a DCF valuation for a ticker with optional overrides.",
            {
                "ticker": {"type": "string"},
                "revenue_growth": {"type": "number"},
                "operating_margin": {"type": "number"},
                "terminal_growth": {"type": "number"},
                "risk_free_rate": {"type": "number"},
                "equity_risk_premium": {"type": "number"},
                "use_exit_multiple": {"type": "boolean"},
                "exit_ev_ebitda_multiple": {"type": "number"},
                "projection_years": {"type": "integer"},
            },
            ["ticker"],
        ),
        fn(
            "valuationengine_run_scenario",
            "Run bull/base/bear DCF scenarios for a ticker.",
            {
                "ticker": {"type": "string"},
                "growth_delta": {"type": "number"},
                "margin_delta": {"type": "number"},
            },
            ["ticker"],
        ),
        fn(
            "valuationengine_run_reverse_dcf",
            "Run reverse DCF for a ticker and field.",
            {
                "ticker": {"type": "string"},
                "field": {"type": "string"},
                "target": {"type": "string"},
            },
            ["ticker"],
        ),
        fn(
            "valuationengine_run_lbo",
            "Run LBO for a ticker with optional overrides.",
            {
                "ticker": {"type": "string"},
                "entry_ev_ebitda_multiple": {"type": "number"},
                "exit_ev_ebitda_multiple": {"type": "number"},
                "debt_pct_purchase": {"type": "number"},
                "interest_rate": {"type": "number"},
                "revenue_growth": {"type": "number"},
                "operating_margin": {"type": "number"},
                "hold_period_years": {"type": "integer"},
            },
            ["ticker"],
        ),
    ]


def _extract_tool_calls(message: dict) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        if item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        name = fn.get("name")
        args_raw = fn.get("arguments") or "{}"
        call_id = item.get("id") or ""
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        if name and call_id:
            calls.append(ToolCall(name=name, arguments=args, call_id=call_id))
    return calls


def _parse_json_content(content: str) -> dict | None:
    if not content:
        return None
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def run_tool_calling_session(
    api_key: str,
    user_prompt: str,
    tool_executor,
    model_preference: list[str] | None = None,
    max_steps: int = 8,
) -> dict:
    """Run OpenAI Chat Completions tool-calling loop."""
    models = model_preference or ["gpt-4o", "gpt-4-turbo"]
    system = (
        "You are a financial analysis assistant. When asked to analyze a stock, "
        "use the valuation engine tools to fetch data and run analyses. "
        "After tools complete, return only structured JSON (no markdown, no prose)."
    )

    last_error: Exception | None = None
    for model in models:
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        try:
            for _step in range(max_steps):
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tool_specs(),
                    "tool_choice": "auto",
                }
                resp = _post_json(OPENAI_CHAT_URL, api_key, payload)
                choice = resp["choices"][0]
                message = choice["message"]
                finish = choice.get("finish_reason")

                tool_calls = _extract_tool_calls(message)
                if tool_calls or finish == "tool_calls":
                    messages.append(message)
                    for call in tool_calls:
                        result = tool_executor(call.name, call.arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": json.dumps(result),
                            }
                        )
                    if not tool_calls:
                        raise OpenAIError(
                            "OpenAI returned a tool-calls finish reason but no tool calls were provided."
                        )
                    continue

                parsed = _parse_json_content(message.get("content") or "")
                if parsed is not None:
                    return parsed

                raise OpenAIError(
                    "OpenAI did not return valid JSON. Try again or disable OpenAI tool-calling."
                )

            raise OpenAIError("OpenAI tool-calling exceeded the maximum number of steps.")
        except OpenAIError as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise OpenAIError("Unable to fetch data. Try again in a moment.")
