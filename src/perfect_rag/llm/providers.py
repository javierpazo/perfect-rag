"""LLM provider implementations."""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (provider-specific)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            If stream=False: Complete response string
            If stream=True: AsyncIterator yielding response chunks
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
        pass

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get model information (pricing, limits, etc.)."""
        return {
            "provider": self.name,
            "model": model,
            "max_tokens": 4096,
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    name = "openai"

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                organization=self.settings.openai_org_id,
                base_url=self.settings.openai_base_url,
            )
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        model = model or self.settings.default_llm_model
        start_time = time.time()

        try:
            if stream:
                return self._stream_response(messages, model, max_tokens, temperature, **kwargs)
            else:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                elapsed = time.time() - start_time
                logger.debug(
                    "OpenAI generation completed",
                    model=model,
                    tokens=response.usage.total_tokens if response.usage else 0,
                    latency_ms=elapsed * 1000,
                )
                return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("OpenAI generation failed", error=str(e), model=model)
            raise

    async def _stream_response(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response chunks."""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    def get_model_info(self, model: str) -> dict[str, Any]:
        # If we're pointing at an OpenAI-compatible proxy (e.g. OpenRouter),
        # many model IDs won't match OpenAI's naming/pricing. Default to 0.0
        # so budget enforcement doesn't block requests due to incorrect estimates.
        is_proxy_model = "/" in model
        is_openrouter = bool(self.settings.openai_base_url and "openrouter.ai" in self.settings.openai_base_url)

        if is_proxy_model and is_openrouter:
            pricing = {"input": 0.0, "output": 0.0}
        else:
            pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})

        return {
            "provider": self.name,
            "model": model,
            "max_tokens": 128000 if "gpt-4" in model else 16384,
            "cost_per_1k_input": pricing["input"],
            "cost_per_1k_output": pricing["output"],
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    name = "anthropic"

    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }

    MODEL_ALIASES = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: AsyncAnthropic | None = None

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return self.MODEL_ALIASES.get(model, model)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        model = self._resolve_model(model or "claude-3.5-sonnet")
        start_time = time.time()

        # Extract system message if present
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        try:
            if stream:
                return self._stream_response(
                    chat_messages, model, max_tokens, temperature, system, **kwargs
                )
            else:
                response = await self.client.messages.create(
                    model=model,
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system or "You are a helpful assistant.",
                    **kwargs,
                )
                elapsed = time.time() - start_time
                logger.debug(
                    "Anthropic generation completed",
                    model=model,
                    tokens=response.usage.input_tokens + response.usage.output_tokens,
                    latency_ms=elapsed * 1000,
                )
                return response.content[0].text

        except Exception as e:
            logger.error("Anthropic generation failed", error=str(e), model=model)
            raise

    async def _stream_response(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response chunks."""
        async with self.client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        try:
            # Simple test message
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
            )
            return True
        except Exception:
            return False

    def get_model_info(self, model: str) -> dict[str, Any]:
        model = self._resolve_model(model)
        pricing = self.PRICING.get(model, {"input": 0.003, "output": 0.015})
        return {
            "provider": self.name,
            "model": model,
            "max_tokens": 200000,
            "cost_per_1k_input": pricing["input"],
            "cost_per_1k_output": pricing["output"],
        }


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._http_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.ollama_url or "http://localhost:11434",
                timeout=120.0,
            )
        return self._http_client

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        model = model or "llama3.2"
        start_time = time.time()

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            if stream:
                return self._stream_response(payload)
            else:
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                elapsed = time.time() - start_time
                logger.debug(
                    "Ollama generation completed",
                    model=model,
                    latency_ms=elapsed * 1000,
                )
                return data["message"]["content"]

        except Exception as e:
            logger.error("Ollama generation failed", error=str(e), model=model)
            raise

    async def _stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream response chunks."""
        async with self.client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    async def health_check(self) -> bool:
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def get_model_info(self, model: str) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": model,
            "max_tokens": 8192,
            "cost_per_1k_input": 0.0,  # Local, no cost
            "cost_per_1k_output": 0.0,
        }
