"""Tests for ``model.custom_headers`` propagation to LLM clients.

Covers the config reader plus each client construction path: the Anthropic
adapter, the auxiliary OpenAI/AsyncOpenAI factory, and the main AIAgent
OpenAI-compatible client.
"""

from types import SimpleNamespace
from unittest.mock import patch

import yaml

from hermes_constants import get_hermes_home
from hermes_cli.config import get_custom_headers
from agent.anthropic_adapter import build_anthropic_client
from agent.auxiliary_client import _build_openai_client, _to_async_client


def _write_model_config(model_section: dict) -> None:
    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": model_section}), encoding="utf-8")


class TestGetCustomHeaders:
    def test_reads_configured_headers(self):
        _write_model_config({
            "default": "anthropic/claude-opus-4.6",
            "custom_headers": {"X-Source": "hermes-agent", "X-Cost-Center": 42},
        })
        assert get_custom_headers() == {
            "X-Source": "hermes-agent",
            "X-Cost-Center": "42",
        }

    def test_missing_returns_empty(self):
        _write_model_config({"default": "gpt-4o"})
        assert get_custom_headers() == {}

    def test_non_dict_model_returns_empty(self):
        config_path = get_hermes_home() / "config.yaml"
        config_path.write_text("model: anthropic/claude-opus-4.6\n", encoding="utf-8")
        assert get_custom_headers() == {}

    def test_drops_blank_names_and_null_values(self):
        _write_model_config({
            "custom_headers": {"": "x", "X-Ok": "yes", "X-Null": None},
        })
        assert get_custom_headers() == {"X-Ok": "yes"}


class TestAnthropicClientCustomHeaders:
    def test_merged_on_top_of_beta_headers(self):
        _write_model_config({"custom_headers": {"X-Source": "hermes-agent"}})
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-something")
            headers = mock_sdk.Anthropic.call_args[1]["default_headers"]
        assert headers["X-Source"] == "hermes-agent"
        assert "interleaved-thinking-2025-05-14" in headers["anthropic-beta"]

    def test_applied_to_third_party_endpoint(self):
        _write_model_config({"custom_headers": {"X-Source": "hermes-agent"}})
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("minimax-secret", base_url="https://api.minimax.io/anthropic")
            headers = mock_sdk.Anthropic.call_args[1]["default_headers"]
        assert headers["X-Source"] == "hermes-agent"
        assert "anthropic-beta" in headers

    def test_absent_config_does_not_change_headers(self):
        _write_model_config({"default": "anthropic/claude-opus-4.6"})
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant-api03-x", base_url="https://custom.api.com"
            )
            headers = mock_sdk.Anthropic.call_args[1]["default_headers"]
        assert headers == {
            "anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
        }


class TestAuxiliaryOpenAICustomHeaders:
    def test_build_openai_client_merges_with_existing_headers(self):
        _write_model_config({"custom_headers": {"X-Source": "hermes-agent"}})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            _build_openai_client(
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://example.com"},
            )
            headers = mock_openai.call_args[1]["default_headers"]
        assert headers["X-Source"] == "hermes-agent"
        assert headers["HTTP-Referer"] == "https://example.com"

    def test_build_openai_client_without_headers_adds_them(self):
        _write_model_config({"custom_headers": {"X-Source": "hermes-agent"}})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            _build_openai_client(api_key="k", base_url="https://api.openai.com/v1")
            headers = mock_openai.call_args[1]["default_headers"]
        assert headers == {"X-Source": "hermes-agent"}

    def test_build_openai_client_noop_when_unset(self):
        _write_model_config({"default": "gpt-4o"})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            _build_openai_client(api_key="k", base_url="https://api.openai.com/v1")
            kwargs = mock_openai.call_args[1]
        assert "default_headers" not in kwargs

    def test_async_client_merges_custom_headers(self):
        _write_model_config({"custom_headers": {"X-Source": "hermes-agent"}})
        sync_client = SimpleNamespace(api_key="k", base_url="https://api.openai.com/v1")
        with patch("openai.AsyncOpenAI") as mock_async:
            _to_async_client(sync_client, "gpt-4o")
            headers = mock_async.call_args[1]["default_headers"]
        assert headers["X-Source"] == "hermes-agent"


class TestAgentOpenAICustomHeaders:
    def _make_agent(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.provider = "openrouter"
        agent.model = "test-model"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._custom_headers = {"X-Source": "hermes-agent"}
        return agent

    def test_merged_into_client_kwargs(self):
        agent = self._make_agent()
        with patch("run_agent.OpenAI") as mock_openai:
            agent._create_openai_client(
                {
                    "api_key": "k",
                    "base_url": "https://openrouter.ai/api/v1",
                    "default_headers": {"HTTP-Referer": "https://example.com"},
                },
                reason="test",
                shared=False,
            )
            headers = mock_openai.call_args[1]["default_headers"]
        assert headers["X-Source"] == "hermes-agent"
        assert headers["HTTP-Referer"] == "https://example.com"

    def test_noop_when_agent_has_no_custom_headers(self):
        agent = self._make_agent()
        agent._custom_headers = {}
        with patch("run_agent.OpenAI") as mock_openai:
            agent._create_openai_client(
                {"api_key": "k", "base_url": "https://openrouter.ai/api/v1"},
                reason="test",
                shared=False,
            )
            kwargs = mock_openai.call_args[1]
        assert "default_headers" not in kwargs
