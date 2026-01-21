"""JWT and API Key authentication."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt

from perfect_rag.auth.models import APIKey, TokenData, User, UserRole
from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class JWTAuth:
    """JWT authentication handler."""

    def __init__(self, settings: Settings | None = None, surrealdb: Any = None):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb  # For database-backed auth (optional)
        self.secret_key = self.settings.jwt_secret_key
        self.algorithm = self.settings.jwt_algorithm
        self.access_token_expire_minutes = self.settings.jwt_expire_minutes

        # In-memory stores (replace with DB in production)
        self._users: dict[str, dict[str, Any]] = {}
        self._api_keys: dict[str, APIKey] = {}
        self._revoked_tokens: set[str] = set()

        # Create default admin user if configured
        if self.settings.admin_username and self.settings.admin_password:
            self._create_default_admin()

    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_id = "admin"
        self._users[admin_id] = {
            "id": admin_id,
            "username": self.settings.admin_username,
            "password_hash": self._hash_password(self.settings.admin_password),
            "email": "admin@localhost",
            "roles": ["admin", "user"],
            "is_active": True,
            "created_at": datetime.utcnow(),
        }
        logger.info("Default admin user created", username=self.settings.admin_username)

    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = self.settings.jwt_secret_key[:16]
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return self._hash_password(password) == password_hash

    def _hash_api_key(self, key: str) -> str:
        """Hash API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_access_token(
        self,
        user_or_id: "User | str",
        roles: list[str] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create JWT access token.

        Args:
            user_or_id: User object or user ID string
            roles: User roles (optional if user object provided)
            expires_delta: Custom expiration time
        """
        # Handle both User object and string user_id
        if isinstance(user_or_id, User):
            user_id = user_or_id.id
            roles = user_or_id.roles
        else:
            user_id = user_or_id
            roles = roles or []

        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire_minutes)

        now = datetime.utcnow()
        expire = now + expires_delta
        jti = secrets.token_hex(16)

        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": expire,
            "iat": now,
            "jti": jti,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> TokenData | None:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                return None

            return TokenData(
                sub=payload.get("sub"),
                roles=payload.get("roles", []),
                exp=datetime.fromtimestamp(payload.get("exp", 0)),
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
                jti=jti,
            )
        except JWTError as e:
            logger.warning("JWT verification failed", error=str(e))
            return None

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI."""
        self._revoked_tokens.add(jti)

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user with username/password."""
        for user_data in self._users.values():
            if user_data["username"] == username:
                if self._verify_password(password, user_data["password_hash"]):
                    return User(
                        id=user_data["id"],
                        username=user_data["username"],
                        email=user_data.get("email"),
                        roles=user_data.get("roles", ["user"]),
                        is_active=user_data.get("is_active", True),
                        created_at=user_data.get("created_at", datetime.utcnow()),
                    )
        return None

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        user_data = self._users.get(user_id)
        if user_data:
            return User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data.get("email"),
                roles=user_data.get("roles", ["user"]),
                is_active=user_data.get("is_active", True),
                created_at=user_data.get("created_at", datetime.utcnow()),
            )
        return None

    def create_user(
        self,
        username: str,
        password: str,
        email: str | None = None,
        roles: list[str] | None = None,
    ) -> User:
        """Create a new user."""
        user_id = secrets.token_hex(8)

        self._users[user_id] = {
            "id": user_id,
            "username": username,
            "password_hash": self._hash_password(password),
            "email": email,
            "roles": roles or ["user"],
            "is_active": True,
            "created_at": datetime.utcnow(),
        }

        return User(
            id=user_id,
            username=username,
            email=email,
            roles=roles or ["user"],
        )

    # =========================================================================
    # API Keys
    # =========================================================================

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
        expires_days: int | None = None,
        roles: list[str] | None = None,
        rate_limit_rpm: int = 60,
    ) -> tuple[str, str]:
        """Create a new API key.

        Returns:
            Tuple of (raw_key, key_id)
        """
        # Generate random API key
        raw_key = f"sk-{secrets.token_hex(24)}"
        key_hash = self._hash_api_key(raw_key)
        key_id = secrets.token_hex(8)

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        api_key = APIKey(
            id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            roles=roles or scopes or ["api"],
            expires_at=expires_at,
            rate_limit_rpm=rate_limit_rpm,
        )

        self._api_keys[key_hash] = api_key

        return raw_key, key_id

    def verify_api_key(self, key: str) -> APIKey | None:
        """Verify an API key."""
        key_hash = self._hash_api_key(key)
        api_key = self._api_keys.get(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()

        return api_key

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by ID."""
        for api_key in self._api_keys.values():
            if api_key.id == key_id:
                api_key.is_active = False
                return True
        return False

    def list_api_keys(self, user_id: str) -> list[APIKey]:
        """List all API keys for a user."""
        return [
            key for key in self._api_keys.values()
            if key.user_id == user_id and key.is_active
        ]


# Global auth instance
_auth: JWTAuth | None = None


def get_auth() -> JWTAuth:
    """Get or create auth instance."""
    global _auth
    if _auth is None:
        _auth = JWTAuth()
    return _auth


def create_access_token(user_id: str, roles: list[str]) -> str:
    """Create access token (convenience function)."""
    return get_auth().create_access_token(user_id, roles)


async def get_current_user(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    api_key: str | None = Depends(api_key_header),
) -> User | None:
    """Get current user from JWT or API key.

    This dependency can be used to optionally authenticate.
    Use get_required_user for required authentication.
    """
    auth = get_auth()

    # Try API key first
    if api_key:
        key_data = auth.verify_api_key(api_key)
        if key_data:
            # Create a virtual user from API key
            return User(
                id=key_data.user_id,
                username=f"api:{key_data.name}",
                roles=key_data.roles,
                rate_limit_rpm=key_data.rate_limit_rpm,
            )

    # Try Bearer token
    if bearer:
        token_data = auth.verify_token(bearer.credentials)
        if token_data:
            user = auth.get_user(token_data.sub)
            if user and user.is_active:
                return user

    # Check for roles header (backward compatibility)
    roles_header = request.headers.get("X-User-Roles", "")
    if roles_header:
        roles = [r.strip() for r in roles_header.split(",") if r.strip()]
        return User(
            id="anonymous",
            username="anonymous",
            roles=roles or ["*"],
        )

    return None


async def get_required_user(
    user: User | None = Depends(get_current_user),
) -> User:
    """Get current user (required).

    Raises 401 if not authenticated.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def verify_api_key(
    api_key: str | None = Depends(api_key_header),
) -> APIKey | None:
    """Verify API key."""
    if not api_key:
        return None
    return get_auth().verify_api_key(api_key)


def require_roles(*required_roles: str):
    """Dependency factory for role-based access control."""
    async def role_checker(user: User = Depends(get_required_user)) -> User:
        user_roles = set(user.roles)
        required = set(required_roles)

        # Admin has all permissions
        if "admin" in user_roles:
            return user

        # Check if user has any required role
        if not user_roles.intersection(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(required_roles)}",
            )

        return user

    return role_checker


# Alias for clarity
get_optional_user = get_current_user
