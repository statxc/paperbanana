"""Tests for the Ollama VLM provider."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from paperbanana.providers.vlm.ollama import OllamaVLM


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Captures the POST payload so tests can inspect it."""

    def __init__(self, response_text: str = "hello"):
        self.captured: dict[str, Any] = {}
        self._response_text = response_text
        self.closed = False

    async def post(self, url: str, json: dict | None = None, **kwargs):
        self.captured["url"] = url
        self.captured["json"] = json
        return _FakeResponse(
            {
                "choices": [{"message": {"content": self._response_text}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    async def aclose(self):
        self.closed = True


@pytest.fixture
def ollama_vlm():
    return OllamaVLM(model="qwen2.5-vl", base_url="http://localhost:11434/v1")


def test_name_and_model(ollama_vlm: OllamaVLM):
    assert ollama_vlm.name == "ollama"
    assert ollama_vlm.model_name == "qwen2.5-vl"


def test_supports_json_mode_default_false(ollama_vlm: OllamaVLM):
    """Ollama defaults to json_mode=False since most open-weight models don't support it."""
    assert ollama_vlm.supports_json_mode is False


def test_supports_json_mode_override():
    vlm = OllamaVLM(model="test", json_mode=True)
    assert vlm.supports_json_mode is True


@pytest.mark.asyncio
async def test_generate_text_only(ollama_vlm: OllamaVLM):
    fake_client = _FakeAsyncClient(response_text="test output")
    ollama_vlm._client = fake_client

    result = await ollama_vlm.generate("Hello")

    assert result == "test output"
    payload = fake_client.captured["json"]
    assert payload["model"] == "qwen2.5-vl"
    assert payload["messages"][-1]["role"] == "user"
    # Text content should be present
    content = payload["messages"][-1]["content"]
    assert any(c["type"] == "text" and c["text"] == "Hello" for c in content)
    # No response_format when json_mode is False
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_generate_with_image(ollama_vlm: OllamaVLM, monkeypatch):
    """Vision images are sent as base64 in the OpenAI multimodal format."""
    monkeypatch.setattr(
        "paperbanana.providers.vlm.ollama.image_to_base64",
        lambda _img: "fake-b64-data",
    )
    fake_client = _FakeAsyncClient(response_text="described the image")
    ollama_vlm._client = fake_client

    img = Image.new("RGB", (4, 4))
    result = await ollama_vlm.generate("Describe this", images=[img])

    assert result == "described the image"
    content = fake_client.captured["json"]["messages"][-1]["content"]
    image_parts = [c for c in content if c["type"] == "image_url"]
    assert len(image_parts) == 1
    assert "fake-b64-data" in image_parts[0]["image_url"]["url"]


@pytest.mark.asyncio
async def test_json_mode_skipped_when_unsupported(ollama_vlm: OllamaVLM):
    """When json_mode=False, response_format is not sent even when requested."""
    fake_client = _FakeAsyncClient(response_text='{"key": "value"}')
    ollama_vlm._client = fake_client

    await ollama_vlm.generate("Return JSON", response_format="json")

    payload = fake_client.captured["json"]
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_json_mode_sent_when_enabled():
    """When json_mode=True, response_format is forwarded."""
    vlm = OllamaVLM(model="test", json_mode=True)
    fake_client = _FakeAsyncClient(response_text='{"key": "value"}')
    vlm._client = fake_client

    await vlm.generate("Return JSON", response_format="json")

    payload = fake_client.captured["json"]
    assert payload["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_system_prompt_included(ollama_vlm: OllamaVLM):
    fake_client = _FakeAsyncClient()
    ollama_vlm._client = fake_client

    await ollama_vlm.generate("Hi", system_prompt="You are helpful.")

    messages = fake_client.captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."


@pytest.mark.asyncio
async def test_close_releases_client(ollama_vlm: OllamaVLM):
    """close() should call aclose on the httpx client and reset it."""
    fake_client = _FakeAsyncClient()
    ollama_vlm._client = fake_client

    await ollama_vlm.close()

    assert fake_client.closed is True
    assert ollama_vlm._client is None


@pytest.mark.asyncio
async def test_close_noop_when_no_client(ollama_vlm: OllamaVLM):
    """close() on a fresh provider (no client created yet) should not error."""
    await ollama_vlm.close()
