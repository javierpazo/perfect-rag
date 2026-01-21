"""Authentication and authorization module."""

from perfect_rag.auth.jwt_auth import (
    JWTAuth,
    create_access_token,
    get_current_user,
    get_optional_user,
    get_required_user,
    verify_api_key,
    require_roles,
)
from perfect_rag.auth.models import User, UserRole, TokenData, APIKey
from perfect_rag.auth.permissions import (
    Permission,
    PermissionAction,
    ResourceType,
    Role,
    RBACManager,
    PermissionChecker,
    DocumentACL,
    get_rbac_manager,
    get_permission_checker,
)

__all__ = [
    # JWT Auth
    "JWTAuth",
    "create_access_token",
    "get_current_user",
    "get_optional_user",
    "get_required_user",
    "verify_api_key",
    "require_roles",
    # Models
    "User",
    "UserRole",
    "TokenData",
    "APIKey",
    # RBAC
    "Permission",
    "PermissionAction",
    "ResourceType",
    "Role",
    "RBACManager",
    "PermissionChecker",
    "DocumentACL",
    "get_rbac_manager",
    "get_permission_checker",
]
