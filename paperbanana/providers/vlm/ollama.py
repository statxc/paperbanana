"""Ollama VLM provider — local open-weight models via the Ollama REST API.

Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint, so it shares
the same message format as the OpenAI provider.  The main differences:
  - No API key required (local server)
  - JSON mode is off by default (most open-weight models don't support it)
  - Vision support depends on the model (e.g. llava, qwen2.5-vl work; llama3 doesn't)
"""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class OllamaVLM(VLMProvider):
    """VLM provider for locally-hosted models via Ollama.

    Talks to Ollama's OpenAI-compatible endpoint. Defaults to
    http://localhost:11434/v1 which is Ollama's standard address.

    Args:
        model: Ollama model name (e.g. 'qwen2.5-vl', 'llava', 'llama3').
        base_url: Ollama server URL. Defaults to localhost.
        json_mode: Whether the model supports response_format: json_object.
            Defaults to False because most open-weight models don't.
    """

    def __init__(
        self,
        model: str = "qwen2.5-vl",
        base_url: str = "http://localhost:11434/v1",
        json_mode: bool = False,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._json_mode = json_mode
        self._client = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=300.0,  # local inference can be slow on CPU
            )
        return self._client

    def is_available(self) -> bool:
        """Ollama doesn't need an API key — just check if the server is reachable."""
        import httpx

        try:
            # Hit the Ollama root endpoint (not /v1) for a quick health check.
            # Strip /v1 suffix to reach the base Ollama server.
            root_url = self._base_url
            if root_url.endswith("/v1"):
                root_url = root_url[:-3]
            resp = httpx.get(root_url, timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=15))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build multimodal content (same format as OpenAI)
        content = []
        if images:
            for img in images:
                b64 = image_to_base64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Only request JSON mode when the model actually supports it.
        if response_format == "json" and self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        logger.debug(
            "Ollama response",
            model=self._model,
            usage=data.get("usage"),
        )
        return text
