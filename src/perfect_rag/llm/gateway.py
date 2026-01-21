"""LLM Gateway with provider selection, fallback, and usage tracking."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, AsyncIterator

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.llm.providers import (
    AnthropicProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
)

logger = structlog.get_logger(__name__)


class UsageTracker:
    """Track LLM usage for cost and budget management."""

    def __init__(self):
        self._usage: dict[str, dict[str, Any]] = {}
        self._daily_reset = datetime.utcnow().date()

    def _get_key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if date changed."""
        today = datetime.utcnow().date()
        if today > self._daily_reset:
            self._usage = {}
            self._daily_reset = today

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Record usage for a generation."""
        self._maybe_reset_daily()
        key = self._get_key(provider, model)

        if key not in self._usage:
            self._usage[key] = {
                "provider": provider,
                "model": model,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "total_latency_ms": 0.0,
                "errors": 0,
            }

        self._usage[key]["requests"] += 1
        self._usage[key]["input_tokens"] += input_tokens
        self._usage[key]["output_tokens"] += output_tokens
        self._usage[key]["total_tokens"] += input_tokens + output_tokens
        self._usage[key]["cost_usd"] += cost_usd
        self._usage[key]["total_latency_ms"] += latency_ms

    def record_error(self, provider: str, model: str) -> None:
        """Record an error."""
        self._maybe_reset_daily()
        key = self._get_key(provider, model)
        if key in self._usage:
            self._usage[key]["errors"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current usage statistics."""
        self._maybe_reset_daily()

        total_cost = sum(u["cost_usd"] for u in self._usage.values())
        total_tokens = sum(u["total_tokens"] for u in self._usage.values())
        total_requests = sum(u["requests"] for u in self._usage.values())

        return {
            "date": str(self._daily_reset),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "by_model": dict(self._usage),
        }

    def get_total_cost_today(self) -> float:
        """Get total cost for today."""
        self._maybe_reset_daily()
        return sum(u["cost_usd"] for u in self._usage.values())


class LLMGateway:
    """Unified gateway for multiple LLM providers with fallback and tracking."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._providers: dict[str, LLMProvider] = {}
        self._usage = UsageTracker()
        self._provider_health: dict[str, bool] = {}
        self._provider_last_check: dict[str, datetime] = {}
        self._health_check_interval = timedelta(minutes=5)

        # Initialize available providers
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize available providers based on configuration."""
        if self.settings.has_openai:
            self._providers["openai"] = OpenAIProvider(self.settings)
            logger.info("Initialized OpenAI provider")

        if self.settings.has_anthropic:
            self._providers["anthropic"] = AnthropicProvider(self.settings)
            logger.info("Initialized Anthropic provider")

        if self.settings.has_ollama:
            self._providers["ollama"] = OllamaProvider(self.settings)
            logger.info("Initialized Ollama provider")

        if not self._providers:
            logger.warning("No LLM providers configured!")

    @property
    def available_providers(self) -> list[str]:
        """List of available provider names."""
        return list(self._providers.keys())

    def get_provider(self, name: str) -> LLMProvider | None:
        """Get a specific provider."""
        return self._providers.get(name)

    async def _check_provider_health(self, name: str) -> bool:
        """Check if provider is healthy (with caching)."""
        now = datetime.utcnow()
        last_check = self._provider_last_check.get(name)

        # Use cached result if recent
        if last_check and (now - last_check) < self._health_check_interval:
            return self._provider_health.get(name, False)

        # Perform health check
        provider = self._providers.get(name)
        if not provider:
            return False

        try:
            healthy = await provider.health_check()
            self._provider_health[name] = healthy
            self._provider_last_check[name] = now
            return healthy
        except Exception:
            self._provider_health[name] = False
            self._provider_last_check[name] = now
            return False

    def _select_provider(
        self,
        preferred: str | None = None,
        fallback_order: list[str] | None = None,
    ) -> tuple[str, LLMProvider] | tuple[None, None]:
        """Select a provider based on preference and availability."""
        if preferred and preferred in self._providers:
            return preferred, self._providers[preferred]

        # Default fallback order
        if fallback_order is None:
            fallback_order = ["openai", "anthropic", "ollama"]

        for name in fallback_order:
            if name in self._providers:
                return name, self._providers[name]

        return None, None

    def _calculate_cost(
        self,
        provider: LLMProvider,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a generation."""
        info = provider.get_model_info(model)
        input_cost = (input_tokens / 1000) * info["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * info["cost_per_1k_output"]
        return input_cost + output_cost

    def _check_budget(self) -> bool:
        """Check if within budget."""
        current_cost = self._usage.get_total_cost_today()
        daily_budget = self.settings.monthly_budget_usd / 30
        return current_cost < daily_budget

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        fallback: bool = True,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """Generate a response using the best available provider.

        Args:
            messages: Chat messages
            model: Model to use (provider-specific)
            provider: Preferred provider name
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            stream: Whether to stream response
            fallback: Whether to try other providers on failure
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text or async iterator of chunks
        """
        # Budget check
        if not self._check_budget():
            raise RuntimeError("Daily budget exceeded")

        # Select provider
        provider_name = provider or self.settings.default_llm_provider
        selected_name, selected_provider = self._select_provider(provider_name)

        if not selected_provider:
            raise RuntimeError("No LLM providers available")

        # Resolve model
        model = model or self.settings.default_llm_model

        # Try generation with fallback
        errors = []
        providers_tried = []

        while selected_provider:
            providers_tried.append(selected_name)

            try:
                start_time = time.time()
                result = await selected_provider.generate(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwargs,
                )

                # If streaming, wrap to track usage after completion
                if stream:
                    return self._wrap_stream(
                        result,
                        selected_name,
                        selected_provider,
                        model,
                        start_time,
                    )

                # Track usage for non-streaming
                elapsed = (time.time() - start_time) * 1000
                # Estimate tokens (rough: 4 chars per token)
                input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
                output_tokens = len(result) // 4
                cost = self._calculate_cost(selected_provider, model, input_tokens, output_tokens)

                self._usage.record(
                    provider=selected_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed,
                    cost_usd=cost,
                )

                return result

            except Exception as e:
                logger.warning(
                    "Provider failed",
                    provider=selected_name,
                    error=str(e),
                )
                errors.append((selected_name, str(e)))
                self._usage.record_error(selected_name, model)

                if not fallback:
                    raise

                # Try next provider
                remaining = [p for p in self._providers if p not in providers_tried]
                if remaining:
                    selected_name, selected_provider = self._select_provider(
                        fallback_order=remaining
                    )
                else:
                    break

        # All providers failed
        error_msg = "; ".join(f"{p}: {e}" for p, e in errors)
        raise RuntimeError(f"All providers failed: {error_msg}")

    async def _wrap_stream(
        self,
        stream: AsyncIterator[str],
        provider_name: str,
        provider: LLMProvider,
        model: str,
        start_time: float,
    ) -> AsyncIterator[str]:
        """Wrap stream to track usage after completion."""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            yield chunk

        # Track usage after stream completes
        elapsed = (time.time() - start_time) * 1000
        full_response = "".join(chunks)
        output_tokens = len(full_response) // 4
        input_tokens = 100  # Estimate, since we don't have access to original messages

        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
        self._usage.record(
            provider=provider_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            cost_usd=cost,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics."""
        return self._usage.get_stats()

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for name in self._providers:
            results[name] = await self._check_provider_health(name)
        return results


# Global gateway instance
_gateway: LLMGateway | None = None


async def get_llm_gateway() -> LLMGateway:
    """Get or create LLM gateway."""
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
