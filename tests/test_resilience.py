"""Tests for resilience patterns (circuit breaker, retry)."""

import asyncio
import pytest

from perfect_rag.core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    RetryConfig,
    RetryWithBackoff,
    ResilientService,
    with_retry,
    with_circuit_breaker,
)


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker):
        """Test circuit starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed
        assert not circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_can_execute_when_closed(self, circuit_breaker):
        """Test requests allowed when circuit is closed."""
        can_execute = await circuit_breaker.can_execute()
        assert can_execute is True

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self, circuit_breaker):
        """Test circuit opens after reaching failure threshold."""
        # Record failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_rejects_requests_when_open(self, circuit_breaker):
        """Test requests rejected when circuit is open."""
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        can_execute = await circuit_breaker.can_execute()
        assert can_execute is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        assert circuit_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.recovery_timeout + 0.1)

        # Try to execute - should transition to half-open
        can_execute = await circuit_breaker.can_execute()
        assert can_execute is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self, circuit_breaker):
        """Test circuit closes after successes in half-open state."""
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.recovery_timeout + 0.1)

        # Transition to half-open
        await circuit_breaker.can_execute()

        # Record enough successes
        for _ in range(circuit_breaker.success_threshold):
            await circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_returns_to_open_on_failure_in_half_open(self, circuit_breaker):
        """Test circuit returns to open on failure in half-open state."""
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        # Wait and transition to half-open
        await asyncio.sleep(circuit_breaker.recovery_timeout + 0.1)
        await circuit_breaker.can_execute()

        # Record a failure
        await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_reset(self, circuit_breaker):
        """Test circuit breaker reset."""
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            await circuit_breaker.record_failure()

        assert circuit_breaker.is_open

        # Reset
        await circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED

    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics."""
        stats = circuit_breaker.get_stats()

        assert "name" in stats
        assert "state" in stats
        assert "failures" in stats
        assert "successes" in stats


class TestRetryWithBackoff:
    """Tests for retry with backoff pattern."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self, retry_config):
        """Test successful execution doesn't retry."""
        retry = RetryWithBackoff(retry_config)
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry.execute(success_func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, retry_config):
        """Test retries on failure."""
        retry = RetryWithBackoff(retry_config)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = await retry.execute(fail_then_succeed)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self, retry_config):
        """Test exhausting all retries."""
        retry = RetryWithBackoff(retry_config)
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")

        with pytest.raises(ValueError):
            await retry.execute(always_fail)

        # Initial + retries
        assert call_count == retry_config.max_retries + 1

    def test_calculate_delay_exponential(self, retry_config):
        """Test exponential backoff delay calculation."""
        retry = RetryWithBackoff(retry_config)

        delay1 = retry.calculate_delay(1)
        delay2 = retry.calculate_delay(2)
        delay3 = retry.calculate_delay(3)

        # Delays should increase exponentially (with some jitter)
        assert delay1 < delay2 < delay3

    def test_calculate_delay_max_cap(self, retry_config):
        """Test delay is capped at max."""
        retry = RetryWithBackoff(retry_config)

        # Very high attempt number
        delay = retry.calculate_delay(100)

        # Should not exceed max_delay + jitter
        assert delay <= retry_config.max_delay * (1 + retry_config.jitter)


class TestResilientService:
    """Tests for resilient service wrapper."""

    @pytest.mark.asyncio
    async def test_protected_decorator_success(self):
        """Test protected decorator on successful function."""
        service = ResilientService("test-service")

        @service.protected
        async def successful_func():
            return "success"

        result = await successful_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_protected_decorator_circuit_opens(self):
        """Test circuit opens after failures with protected decorator."""
        service = ResilientService(
            "test-service",
            circuit_breaker=CircuitBreaker(
                "test",
                failure_threshold=2,
                recovery_timeout=10.0,
            ),
            retry_config=RetryConfig(max_retries=0),  # No retries
        )

        call_count = 0

        @service.protected
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")

        # First two calls should fail but go through
        for _ in range(2):
            with pytest.raises(ValueError):
                await failing_func()

        # Third call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpen):
            await failing_func()

        assert call_count == 2  # Third call was blocked

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test protect context manager on success."""
        service = ResilientService("test-service")

        async with service.protect():
            result = "success"

        assert result == "success"

    @pytest.mark.asyncio
    async def test_context_manager_records_failure(self):
        """Test protect context manager records failure."""
        service = ResilientService(
            "test-service",
            circuit_breaker=CircuitBreaker(
                "test",
                failure_threshold=2,
            ),
        )

        try:
            async with service.protect():
                raise ValueError("Error")
        except ValueError:
            pass

        stats = service.get_stats()
        assert stats["circuit_breaker"]["failures"] == 1


class TestDecorators:
    """Tests for convenience decorators."""

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test @with_retry decorator."""
        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_decorator(self):
        """Test @with_circuit_breaker decorator."""
        call_count = 0

        @with_circuit_breaker("test-cb", failure_threshold=2, recovery_timeout=10.0)
        async def protected_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")

        # First two calls fail normally
        for _ in range(2):
            with pytest.raises(ValueError):
                await protected_func()

        # Third call blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpen):
            await protected_func()

        assert call_count == 2
