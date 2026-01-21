"""Authentication models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"  # For API key access


class TokenData(BaseModel):
    """JWT token payload."""

    sub: str  # User ID
    roles: list[str] = Field(default_factory=list)
    exp: datetime | None = None
    iat: datetime | None = None
    jti: str | None = None  # JWT ID for revocation


class User(BaseModel):
    """User model."""

    id: str
    username: str
    email: str | None = None
    roles: list[str] = Field(default_factory=lambda: ["user"])
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Rate limiting
    rate_limit_rpm: int = Field(default=60, description="Requests per minute")
    rate_limit_tpm: int = Field(default=100000, description="Tokens per minute")

    # Budget
    monthly_budget_usd: float | None = None


class APIKey(BaseModel):
    """API Key model."""

    id: str
    key_hash: str  # Hashed API key
    name: str
    user_id: str
    roles: list[str] = Field(default_factory=lambda: ["api"])
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None

    # Rate limiting
    rate_limit_rpm: int = Field(default=60)
    rate_limit_tpm: int = Field(default=100000)


class LoginRequest(BaseModel):
    """Login request."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class CreateAPIKeyRequest(BaseModel):
    """Request to create API key."""

    name: str
    expires_in_days: int | None = None
    roles: list[str] | None = None
    rate_limit_rpm: int = 60


class APIKeyResponse(BaseModel):
    """API key creation response."""

    id: str
    key: str  # Only returned on creation
    name: str
    created_at: datetime
    expires_at: datetime | None = None
