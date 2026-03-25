"""Tests for open-weight VLM support: extract_json, capability checks, registry."""

from __future__ import annotations

import json

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.types import ReferenceExample
from paperbanana.core.utils import extract_json
from paperbanana.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# extract_json — robust JSON extraction from free-form VLM output
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_plain_json_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_json_in_markdown_fence(self):
        text = 'Here is the result:\n```json\n{"selected_ids": ["a", "b"]}\n```\nDone.'
        result = extract_json(text)
        assert result == {"selected_ids": ["a", "b"]}

    def test_json_in_bare_fence(self):
        text = 'Sure:\n```\n{"winner": "Model"}\n```'
        result = extract_json(text)
        assert result == {"winner": "Model"}

    def test_json_embedded_in_text(self):
        text = 'I think the answer is {"score": 42, "label": "good"} based on my analysis.'
        result = extract_json(text)
        assert result == {"score": 42, "label": "good"}

    def test_nested_braces(self):
        obj = {"outer": {"inner": [1, 2]}, "key": "val"}
        text = f"Result: {json.dumps(obj)}"
        result = extract_json(text)
        assert result == obj

    def test_array_embedded_in_text(self):
        text = 'The IDs are: ["ref_001", "ref_002"]. That is my selection.'
        result = extract_json(text)
        assert result == ["ref_001", "ref_002"]

    def test_returns_none_for_no_json(self):
        assert extract_json("This is just plain text with no JSON.") is None

    def test_returns_none_for_empty_string(self):
        assert extract_json("") is None

    def test_handles_strings_with_braces(self):
        """Strings containing braces inside JSON values shouldn't break parsing."""
        obj = {"text": "use {curly} braces"}
        text = json.dumps(obj)
        result = extract_json(text)
        assert result == obj

    def test_whitespace_around_json(self):
        text = '  \n  {"key": "value"}  \n  '
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_truncated_json_returns_none(self):
        text = '{"key": "value", "incomplete'
        assert extract_json(text) is None


# ---------------------------------------------------------------------------
# VLMProvider.supports_json_mode — default and overrides
# ---------------------------------------------------------------------------


class TestSupportsJsonMode:
    def test_gemini_supports_json(self):
        settings = Settings(vlm_provider="gemini", google_api_key="test-key")
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is True

    def test_openai_supports_json_by_default(self):
        settings = Settings(vlm_provider="openai", openai_api_key="test-key")
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is True

    def test_openai_local_defaults_no_json(self):
        settings = Settings(
            vlm_provider="openai_local",
            openai_base_url="http://localhost:8000/v1",
        )
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is False
        assert vlm.name == "openai"

    def test_openai_local_json_mode_override(self):
        settings = Settings(
            vlm_provider="openai_local",
            openai_base_url="http://localhost:8000/v1",
            openai_local_json_mode=True,
        )
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is True

    def test_ollama_defaults_no_json(self):
        settings = Settings(vlm_provider="ollama")
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is False

    def test_ollama_json_mode_override(self):
        settings = Settings(vlm_provider="ollama", ollama_json_mode=True)
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.supports_json_mode is True


# ---------------------------------------------------------------------------
# Registry — Ollama and openai_local creation
# ---------------------------------------------------------------------------


class TestRegistryLocalProviders:
    def test_create_ollama_vlm(self):
        settings = Settings(
            vlm_provider="ollama",
            vlm_model="llava",
            ollama_base_url="http://myserver:11434/v1",
        )
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.name == "ollama"
        assert vlm.model_name == "llava"

    def test_ollama_model_override(self):
        settings = Settings(
            vlm_provider="ollama",
            vlm_model="default-model",
            ollama_model="qwen2.5-vl:72b",
        )
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.model_name == "qwen2.5-vl:72b"

    def test_create_openai_local_vlm(self):
        settings = Settings(
            vlm_provider="openai_local",
            vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
            openai_base_url="http://localhost:8000/v1",
        )
        vlm = ProviderRegistry.create_vlm(settings)
        assert vlm.name == "openai"
        assert vlm.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_unknown_provider_error_message_includes_new_providers(self):
        settings = Settings(vlm_provider="nonexistent")
        with pytest.raises(ValueError, match="ollama"):
            ProviderRegistry.create_vlm(settings)


# ---------------------------------------------------------------------------
# Agent integration — retriever/critic respect supports_json_mode
# ---------------------------------------------------------------------------


class _MockVLM:
    """Mock VLM with configurable json_mode support."""

    name = "mock"
    model_name = "mock-model"

    def __init__(self, response: str = "", json_mode: bool = True):
        self._response = response
        self._json_mode = json_mode
        self.last_response_format = "NOT_CALLED"

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    async def generate(
        self,
        prompt,
        images=None,
        system_prompt=None,
        temperature=1.0,
        max_tokens=4096,
        response_format=None,
    ):
        self.last_response_format = response_format
        return self._response


def _make_examples(n: int) -> list[ReferenceExample]:
    return [
        ReferenceExample(
            id=f"ref_{i:03d}",
            source_context=f"Context {i}",
            caption=f"Caption {i}",
            image_path=f"images/{i}.png",
        )
        for i in range(n)
    ]


class TestRetrieverJsonMode:
    @pytest.mark.asyncio
    async def test_skips_json_format_when_unsupported(self):
        from paperbanana.agents.retriever import RetrieverAgent

        # Model returns JSON wrapped in markdown (common for open-weight models)
        response = '```json\n{"selected_ids": ["ref_001"]}\n```'
        vlm = _MockVLM(response=response, json_mode=False)
        agent = RetrieverAgent(vlm)

        result = await agent.run(
            source_context="test",
            caption="test",
            candidates=_make_examples(5),
            num_examples=2,
        )

        # Should NOT have sent response_format="json"
        assert vlm.last_response_format is None
        # Should still parse the fenced JSON correctly
        assert len(result) == 1
        assert result[0].id == "ref_001"

    @pytest.mark.asyncio
    async def test_sends_json_format_when_supported(self):
        from paperbanana.agents.retriever import RetrieverAgent

        response = json.dumps({"selected_ids": ["ref_000"]})
        vlm = _MockVLM(response=response, json_mode=True)
        agent = RetrieverAgent(vlm)

        await agent.run(
            source_context="test",
            caption="test",
            candidates=_make_examples(5),
            num_examples=2,
        )

        assert vlm.last_response_format == "json"


class TestCriticJsonMode:
    @pytest.mark.asyncio
    async def test_skips_json_format_when_unsupported(self, tmp_path):
        from paperbanana.agents.critic import CriticAgent

        # Critic response with markdown fences
        response = '```json\n{"critic_suggestions": ["fix colors"]}\n```'
        vlm = _MockVLM(response=response, json_mode=False)
        agent = CriticAgent(vlm)

        # Create a tiny test image
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.new("RGB", (4, 4)).save(img_path)

        # Need a prompt file for the critic
        prompt_dir = tmp_path / "diagram"
        prompt_dir.mkdir()
        (prompt_dir / "critic.txt").write_text(
            "Evaluate: {source_context}\nCaption: {caption}\nDescription: {description}"
        )
        agent.prompt_dir = tmp_path

        result = await agent.run(
            image_path=str(img_path),
            description="test description",
            source_context="test context",
            caption="test caption",
        )

        assert vlm.last_response_format is None
        assert result.needs_revision is True
        assert "fix colors" in result.critic_suggestions
