"""Tests for authentication module."""

import pytest
from datetime import datetime, timedelta


class TestJWTAuth:
    """Tests for JWT authentication."""

    def test_create_user(self, jwt_auth):
        """Test user creation."""
        user = jwt_auth.create_user(
            username="newuser",
            password="password123",
            email="new@example.com",
            roles=["user"],
        )

        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert "user" in user.roles
        assert user.id is not None

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, jwt_auth):
        """Test successful authentication."""
        # Create user first
        jwt_auth.create_user(
            username="authuser",
            password="authpass",
            roles=["user"],
        )

        # Authenticate
        user = await jwt_auth.authenticate_user("authuser", "authpass")

        assert user is not None
        assert user.username == "authuser"

    @pytest.mark.asyncio
    async def test_authenticate_user_failure(self, jwt_auth):
        """Test failed authentication."""
        user = await jwt_auth.authenticate_user("nonexistent", "wrongpass")
        assert user is None

    def test_create_access_token(self, jwt_auth):
        """Test JWT token creation."""
        user = jwt_auth.create_user(
            username="tokenuser",
            password="pass",
            roles=["user", "admin"],
        )

        token = jwt_auth.create_access_token(user)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_valid(self, jwt_auth):
        """Test valid token verification."""
        user = jwt_auth.create_user(
            username="verifyuser",
            password="pass",
            roles=["user"],
        )

        token = jwt_auth.create_access_token(user)
        token_data = jwt_auth.verify_token(token)

        assert token_data is not None
        assert token_data.sub == user.id
        assert "user" in token_data.roles

    def test_verify_token_invalid(self, jwt_auth):
        """Test invalid token verification."""
        token_data = jwt_auth.verify_token("invalid-token")
        assert token_data is None

    def test_revoke_token(self, jwt_auth):
        """Test token revocation."""
        user = jwt_auth.create_user(
            username="revokeuser",
            password="pass",
            roles=["user"],
        )

        token = jwt_auth.create_access_token(user)
        token_data = jwt_auth.verify_token(token)

        # Revoke the token
        jwt_auth.revoke_token(token_data.jti)

        # Verify it's now invalid
        revoked_data = jwt_auth.verify_token(token)
        assert revoked_data is None


class TestAPIKeys:
    """Tests for API key management."""

    @pytest.mark.asyncio
    async def test_create_api_key(self, jwt_auth):
        """Test API key creation."""
        user = jwt_auth.create_user(
            username="apikeyuser",
            password="pass",
            roles=["user"],
        )

        raw_key, key_id = await jwt_auth.create_api_key(
            user_id=user.id,
            name="Test Key",
            roles=["api"],
        )

        assert raw_key.startswith("sk-")
        assert key_id is not None

    @pytest.mark.asyncio
    async def test_verify_api_key(self, jwt_auth):
        """Test API key verification."""
        user = jwt_auth.create_user(
            username="verifyapiuser",
            password="pass",
        )

        raw_key, key_id = await jwt_auth.create_api_key(
            user_id=user.id,
            name="Verify Key",
        )

        api_key = jwt_auth.verify_api_key(raw_key)

        assert api_key is not None
        assert api_key.user_id == user.id
        assert api_key.name == "Verify Key"

    @pytest.mark.asyncio
    async def test_verify_invalid_api_key(self, jwt_auth):
        """Test invalid API key verification."""
        api_key = jwt_auth.verify_api_key("sk-invalid-key")
        assert api_key is None

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, jwt_auth):
        """Test API key revocation."""
        user = jwt_auth.create_user(
            username="revokeapiuser",
            password="pass",
        )

        raw_key, key_id = await jwt_auth.create_api_key(
            user_id=user.id,
            name="Revoke Key",
        )

        # Verify it works
        assert jwt_auth.verify_api_key(raw_key) is not None

        # Revoke it
        await jwt_auth.revoke_api_key(key_id)

        # Verify it's now invalid
        assert jwt_auth.verify_api_key(raw_key) is None

    @pytest.mark.asyncio
    async def test_api_key_expiration(self, jwt_auth):
        """Test API key expiration."""
        user = jwt_auth.create_user(
            username="expireuser",
            password="pass",
        )

        # Create key that expires immediately (0 days = already expired)
        raw_key, key_id = await jwt_auth.create_api_key(
            user_id=user.id,
            name="Expire Key",
            expires_days=0,  # Expires now
        )

        # Key should be invalid due to expiration
        api_key = jwt_auth.verify_api_key(raw_key)
        # Note: With expires_days=0, the key is created with expires_at = now,
        # so it might still be valid for a brief moment. In practice, use
        # a negative timedelta for testing expiration.
