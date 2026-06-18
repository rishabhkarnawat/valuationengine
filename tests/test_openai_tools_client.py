"""Tests for OpenAI tool-calling client (mocked HTTP)."""

import json
from unittest.mock import patch

import pytest

from valuationengine.adapters import openai_tools_client as oai


def _http_response(payload: dict):
  class Resp:
    def read(self):
      return json.dumps(payload).encode("utf-8")

    def __enter__(self):
      return self

    def __exit__(self, *args):
      return False

  return Resp()


def test_tool_calling_loop_executes_tools_and_returns_json():
  first = {
    "choices": [
      {
        "finish_reason": "tool_calls",
        "message": {
          "role": "assistant",
          "content": None,
          "tool_calls": [
            {
              "id": "call_1",
              "type": "function",
              "function": {
                "name": "valuationengine_fetch_fundamentals",
                "arguments": json.dumps({"ticker": "DPZ"}),
              },
            }
          ],
        },
      }
    ]
  }
  second = {
    "choices": [
      {
        "finish_reason": "stop",
        "message": {
          "role": "assistant",
          "content": json.dumps({"fundamentals": {"ticker": "DPZ"}}),
        },
      }
    ]
  }

  calls = []

  def executor(name, args):
    calls.append((name, args))
    return {"ticker": "DPZ"}

  with patch("urllib.request.urlopen", side_effect=[_http_response(first), _http_response(second)]):
    result = oai.run_tool_calling_session("sk-test", "analyze DPZ", executor)

  assert result == {"fundamentals": {"ticker": "DPZ"}}
  assert calls == [("valuationengine_fetch_fundamentals", {"ticker": "DPZ"})]


def test_invalid_api_key_raises_clear_error():
  import io
  import urllib.error

  err = urllib.error.HTTPError(
    url="https://api.openai.com/v1/chat/completions",
    code=401,
    msg="Unauthorized",
    hdrs=None,
    fp=io.BytesIO(b'{"error":{"message":"Incorrect API key provided"}}'),
  )

  with patch("urllib.request.urlopen", side_effect=err):
    with pytest.raises(oai.OpenAIError, match="invalid or missing"):
      oai.run_tool_calling_session("bad", "hi", lambda n, a: {})


def test_get_api_key_whitespace_falls_back_to_env(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
  assert oai.get_api_key("   ") == "sk-env-key"
  assert oai.get_api_key("sk-explicit") == "sk-explicit"
