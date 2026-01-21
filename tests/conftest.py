"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment
os.environ["SURREALDB_URL"] = "memory"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only-32chars"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "testadmin"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    """Get test settings."""
    from perfect_rag.config import Settings
    return Settings(
        surrealdb_url="memory",
        jwt_secret_key="test-secret-key-for-testing-only-32chars",
        admin_username="admin",
        admin_password="testadmin",
    )


@pytest.fixture
def jwt_auth(settings):
    """Create JWT auth instance for testing."""
    from perfect_rag.auth import JWTAuth
    return JWTAuth(settings=settings)


@pytest.fixture
def test_user(jwt_auth):
    """Create a test user."""
    return jwt_auth.create_user(
        username="testuser",
        password="testpass",
        email="test@example.com",
        roles=["user"],
    )


@pytest.fixture
def admin_user(jwt_auth):
    """Get admin user."""
    user = jwt_auth.authenticate_user("admin", "testadmin")
    return user


@pytest.fixture
def hallucination_detector(settings):
    """Create hallucination detector for testing."""
    from perfect_rag.generation.hallucination_detector import HallucinationDetector
    return HallucinationDetector(settings=settings)


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker for testing."""
    from perfect_rag.core.resilience import CircuitBreaker
    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout=1.0,
    )


@pytest.fixture
def retry_config():
    """Create retry config for testing."""
    from perfect_rag.core.resilience import RetryConfig
    return RetryConfig(
        max_retries=3,
        initial_delay=0.1,
        max_delay=1.0,
    )


# =============================================================================
# Mock fixtures for external services
# =============================================================================


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    async def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding."""
        import hashlib
        # Generate deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes]
        # Pad or truncate to dimension
        while len(embedding) < self.dimension:
            embedding.extend(embedding)
        return embedding[:self.dimension]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        return [await self.embed_query(t) for t in texts]

    async def health_check(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {"model": "mock", "dimension": self.dimension}


@pytest.fixture
def mock_embedding_service():
    """Get mock embedding service."""
    return MockEmbeddingService()


class MockLLMGateway:
    """Mock LLM gateway for testing."""

    async def generate(
        self,
        messages: list[dict],
        model: str = "mock",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        """Generate mock response."""
        if stream:
            async def _stream():
                yield "This is a "
                yield "mock response "
                yield "for testing."
            return _stream()

        return "This is a mock response for testing."

    async def health_check(self) -> dict:
        return {"mock": True}

    @property
    def available_providers(self) -> list[str]:
        return ["mock"]

    def get_usage_stats(self) -> dict:
        return {"requests": 0, "tokens": 0}


@pytest.fixture
def mock_llm_gateway():
    """Get mock LLM gateway."""
    return MockLLMGateway()


class MockSurrealDB:
    """Mock SurrealDB client for testing."""

    def __init__(self):
        self._data = {}

    async def query(self, query: str, params: dict = None):
        """Execute mock query."""
        return [{"result": []}]

    async def health_check(self) -> bool:
        return True

    async def disconnect(self):
        pass

    async def list_documents(self, limit: int = 100, offset: int = 0):
        return []

    async def get_document(self, doc_id: str):
        return None

    async def get_chunks_by_ids(self, chunk_ids: list[str]):
        return []


@pytest.fixture
def mock_surrealdb():
    """Get mock SurrealDB client."""
    return MockSurrealDB()
