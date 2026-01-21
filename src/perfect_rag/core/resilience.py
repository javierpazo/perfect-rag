"""Circuit breakers and retry patterns for resilient operations."""

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    failures: int = 0
    successes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service is down.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered, allow limited requests
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
    ):
        """
        Args:
            name: Circuit breaker identifier
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls in half-open state
            success_threshold: Successes needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: datetime | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    async def can_execute(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._opened_at and datetime.utcnow() > self._opened_at + timedelta(
                    seconds=self.recovery_timeout
                ):
                    self._transition_to_half_open()
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.successes += 1
            self._stats.last_success_time = datetime.utcnow()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to_closed()

    async def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failures += 1
            self._stats.last_failure_time = datetime.utcnow()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to_open()

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._opened_at = datetime.utcnow()
        logger.warning(
            "Circuit breaker opened",
            name=self.name,
            failures=self._stats.consecutive_failures,
        )

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._stats.consecutive_successes = 0
        logger.info("Circuit breaker half-open", name=self.name)

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_calls = 0
        self._stats.consecutive_failures = 0
        logger.info("Circuit breaker closed", name=self.name)

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._stats.failures,
            "successes": self._stats.successes,
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "last_failure": (
                self._stats.last_failure_time.isoformat()
                if self._stats.last_failure_time
                else None
            ),
            "last_success": (
                self._stats.last_success_time.isoformat()
                if self._stats.last_success_time
                else None
            ),
        }

    async def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._opened_at = None
            self._half_open_calls = 0


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, name: str, message: str = "Circuit breaker is open"):
        self.name = name
        super().__init__(f"{message}: {name}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )


class RetryWithBackoff:
    """Retry operations with exponential backoff and jitter.

    Features:
    - Exponential backoff with configurable base
    - Random jitter to prevent thundering herd
    - Configurable max retries and delays
    - Exception filtering
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt with jitter."""
        # Exponential backoff
        delay = self.config.initial_delay * (
            self.config.exponential_base ** (attempt - 1)
        )

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter = delay * self.config.jitter * random.random()
        delay = delay + jitter

        return delay

    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        return isinstance(exception, self.config.retryable_exceptions)

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(1, self.config.max_retries + 2):  # +1 for initial + retries
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.is_retryable(e):
                    logger.warning(
                        "Non-retryable exception",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

                if attempt > self.config.max_retries:
                    logger.error(
                        "All retries exhausted",
                        attempts=attempt,
                        error=str(e),
                    )
                    raise

                delay = self.calculate_delay(attempt)
                logger.warning(
                    "Retrying after failure",
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    delay=delay,
                    error=str(e),
                )

                await asyncio.sleep(delay)

        # Should never reach here, but for type safety
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in retry logic")


class ResilientService:
    """Combine circuit breaker and retry for resilient service calls.

    Usage:
        service = ResilientService("external_api")

        @service.protected
        async def call_external_api():
            ...

        # Or use context manager
        async with service.protect():
            result = await call_external_api()
    """

    def __init__(
        self,
        name: str,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name)
        self.retry = RetryWithBackoff(retry_config)

    def protected(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect async functions with circuit breaker and retry."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not await self.circuit_breaker.can_execute():
                raise CircuitBreakerOpen(self.name)

            try:
                result = await self.retry.execute(func, *args, **kwargs)
                await self.circuit_breaker.record_success()
                return result
            except Exception as e:
                await self.circuit_breaker.record_failure(e)
                raise

        return wrapper

    class _ProtectContext:
        """Context manager for protected execution."""

        def __init__(self, service: "ResilientService"):
            self.service = service

        async def __aenter__(self) -> "ResilientService._ProtectContext":
            if not await self.service.circuit_breaker.can_execute():
                raise CircuitBreakerOpen(self.service.name)
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: Any,
        ) -> bool:
            if exc_type is None:
                await self.service.circuit_breaker.record_success()
            else:
                await self.service.circuit_breaker.record_failure(
                    exc_val if isinstance(exc_val, Exception) else None
                )
            return False

    def protect(self) -> _ProtectContext:
        """Context manager for protecting a block of code."""
        return self._ProtectContext(self)

    def get_stats(self) -> dict[str, Any]:
        """Get service resilience statistics."""
        return {
            "name": self.name,
            "circuit_breaker": self.circuit_breaker.get_stats(),
        }


# =============================================================================
# Registry for managing multiple circuit breakers
# =============================================================================


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""

    _instance: "CircuitBreakerRegistry | None" = None
    _breakers: dict[str, CircuitBreaker]

    def __new__(cls) -> "CircuitBreakerRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._breakers = {}
        return cls._instance

    def get_or_create(
        self,
        name: str,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, **kwargs)
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            await cb.reset()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return CircuitBreakerRegistry()


# =============================================================================
# Convenience decorators
# =============================================================================


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry with backoff to async functions.

    Usage:
        @with_retry(max_retries=3)
        async def flaky_operation():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
    )
    retry = RetryWithBackoff(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add circuit breaker to async functions.

    Usage:
        @with_circuit_breaker("external_api")
        async def call_external():
            ...
    """
    registry = get_circuit_breaker_registry()
    breaker = registry.get_or_create(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not await breaker.can_execute():
                raise CircuitBreakerOpen(name)

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                await breaker.record_success()
                return result
            except Exception as e:
                await breaker.record_failure(e)
                raise

        return wrapper

    return decorator
