"""Role-Based Access Control (RBAC) for RAG systems.

Provides granular permissions, role management, and document-level
access control for secure multi-tenant RAG deployments.
"""

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class PermissionAction(str, Enum):
    """Permission actions."""

    # Document operations
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"

    # Query operations
    QUERY = "query"
    QUERY_WITH_CONTEXT = "query_with_context"

    # Admin operations
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_DOCUMENTS = "manage_documents"
    MANAGE_SOURCES = "manage_sources"
    VIEW_ANALYTICS = "view_analytics"
    VIEW_AUDIT_LOGS = "view_audit_logs"

    # System operations
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_API_KEYS = "manage_api_keys"


class ResourceType(str, Enum):
    """Types of resources that can be protected."""

    DOCUMENT = "document"
    COLLECTION = "collection"
    CHUNK = "chunk"
    QUERY = "query"
    USER = "user"
    ROLE = "role"
    API_KEY = "api_key"
    SOURCE = "source"
    SYSTEM = "system"


@dataclass
class Permission:
    """A permission definition."""

    permission_id: str
    action: PermissionAction
    resource_type: ResourceType
    resource_id: str | None = None  # None = all resources of type
    conditions: dict[str, Any] = field(default_factory=dict)  # Additional conditions
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "permission_id": self.permission_id,
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "conditions": self.conditions,
            "description": self.description,
        }

    def matches(
        self,
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: str | None = None,
    ) -> bool:
        """Check if this permission matches the requested access."""
        if self.action != action:
            return False
        if self.resource_type != resource_type:
            return False
        # None means all resources of this type
        if self.resource_id is not None and self.resource_id != resource_id:
            return False
        return True


class Role(BaseModel):
    """A role with a set of permissions."""

    role_id: str = Field(..., description="Unique role identifier")
    name: str = Field(..., description="Human-readable role name")
    description: str = Field(default="")
    permissions: list[str] = Field(
        default_factory=list,
        description="List of permission IDs",
    )
    parent_roles: list[str] = Field(
        default_factory=list,
        description="Parent roles (for inheritance)",
    )
    is_system: bool = Field(
        default=False,
        description="System roles cannot be modified",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


@dataclass
class DocumentACL:
    """Document-level access control list."""

    document_id: str
    owner_id: str
    read_users: list[str] = field(default_factory=list)
    write_users: list[str] = field(default_factory=list)
    read_roles: list[str] = field(default_factory=list)
    write_roles: list[str] = field(default_factory=list)
    is_public: bool = False
    inherit_collection_acl: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "owner_id": self.owner_id,
            "read_users": self.read_users,
            "write_users": self.write_users,
            "read_roles": self.read_roles,
            "write_roles": self.write_roles,
            "is_public": self.is_public,
            "inherit_collection_acl": self.inherit_collection_acl,
        }

    def can_read(self, user_id: str, user_roles: list[str]) -> bool:
        """Check if user can read this document."""
        if self.is_public:
            return True
        if user_id == self.owner_id:
            return True
        if user_id in self.read_users or user_id in self.write_users:
            return True
        if any(role in self.read_roles or role in self.write_roles for role in user_roles):
            return True
        return False

    def can_write(self, user_id: str, user_roles: list[str]) -> bool:
        """Check if user can write this document."""
        if user_id == self.owner_id:
            return True
        if user_id in self.write_users:
            return True
        if any(role in self.write_roles for role in user_roles):
            return True
        return False


class PermissionStore:
    """Storage for permissions and roles."""

    def __init__(self, surrealdb_client: Any = None):
        self._permissions: dict[str, Permission] = {}
        self._roles: dict[str, Role] = {}
        self._user_roles: dict[str, list[str]] = {}  # user_id -> role_ids
        self._document_acls: dict[str, DocumentACL] = {}
        self._lock = asyncio.Lock()
        self.surrealdb = surrealdb_client

    async def add_permission(self, permission: Permission) -> None:
        """Add a permission."""
        async with self._lock:
            self._permissions[permission.permission_id] = permission

    async def get_permission(self, permission_id: str) -> Permission | None:
        """Get a permission by ID."""
        return self._permissions.get(permission_id)

    async def add_role(self, role: Role) -> None:
        """Add a role."""
        async with self._lock:
            self._roles[role.role_id] = role

        if self.surrealdb:
            await self._store_role(role)

    async def get_role(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        return self._roles.get(role_id)

    async def list_roles(self) -> list[Role]:
        """List all roles."""
        return list(self._roles.values())

    async def update_role(self, role: Role) -> None:
        """Update a role."""
        role.updated_at = datetime.utcnow()
        async with self._lock:
            self._roles[role.role_id] = role

        if self.surrealdb:
            await self._store_role(role)

    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        async with self._lock:
            if role_id in self._roles:
                role = self._roles[role_id]
                if role.is_system:
                    return False
                del self._roles[role_id]
                return True
        return False

    async def assign_role(self, user_id: str, role_id: str) -> None:
        """Assign a role to a user."""
        async with self._lock:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = []
            if role_id not in self._user_roles[user_id]:
                self._user_roles[user_id].append(role_id)

    async def revoke_role(self, user_id: str, role_id: str) -> None:
        """Revoke a role from a user."""
        async with self._lock:
            if user_id in self._user_roles:
                if role_id in self._user_roles[user_id]:
                    self._user_roles[user_id].remove(role_id)

    async def get_user_roles(self, user_id: str) -> list[str]:
        """Get roles assigned to a user."""
        return self._user_roles.get(user_id, [])

    async def set_document_acl(self, acl: DocumentACL) -> None:
        """Set ACL for a document."""
        async with self._lock:
            self._document_acls[acl.document_id] = acl

    async def get_document_acl(self, document_id: str) -> DocumentACL | None:
        """Get ACL for a document."""
        return self._document_acls.get(document_id)

    async def _store_role(self, role: Role) -> None:
        """Store role in database."""
        if not self.surrealdb:
            return

        try:
            await self.surrealdb.client.query(
                """
                UPSERT role SET
                    role_id = $role_id,
                    name = $name,
                    description = $description,
                    permissions = $permissions,
                    parent_roles = $parent_roles,
                    is_system = $is_system,
                    created_at = $created_at,
                    updated_at = $updated_at,
                    metadata = $metadata
                WHERE role_id = $role_id
                """,
                role.model_dump(),
            )
        except Exception as e:
            logger.error("Failed to store role", error=str(e))


class RBACManager:
    """Manage roles and permissions.

    Provides role hierarchy, permission inheritance, and role management.
    """

    def __init__(
        self,
        store: PermissionStore | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or PermissionStore()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize with default roles and permissions."""
        if self._initialized:
            return

        # Create default permissions
        await self._create_default_permissions()

        # Create default roles
        await self._create_default_roles()

        self._initialized = True
        logger.info("RBAC manager initialized")

    async def _create_default_permissions(self) -> None:
        """Create default permissions."""
        permissions = [
            # Document permissions
            Permission(
                permission_id="document:read",
                action=PermissionAction.READ,
                resource_type=ResourceType.DOCUMENT,
                description="Read documents",
            ),
            Permission(
                permission_id="document:write",
                action=PermissionAction.WRITE,
                resource_type=ResourceType.DOCUMENT,
                description="Create and update documents",
            ),
            Permission(
                permission_id="document:delete",
                action=PermissionAction.DELETE,
                resource_type=ResourceType.DOCUMENT,
                description="Delete documents",
            ),
            Permission(
                permission_id="document:share",
                action=PermissionAction.SHARE,
                resource_type=ResourceType.DOCUMENT,
                description="Share documents with others",
            ),
            # Query permissions
            Permission(
                permission_id="query:execute",
                action=PermissionAction.QUERY,
                resource_type=ResourceType.QUERY,
                description="Execute queries",
            ),
            Permission(
                permission_id="query:with_context",
                action=PermissionAction.QUERY_WITH_CONTEXT,
                resource_type=ResourceType.QUERY,
                description="Execute queries with full context",
            ),
            # Admin permissions
            Permission(
                permission_id="users:manage",
                action=PermissionAction.MANAGE_USERS,
                resource_type=ResourceType.USER,
                description="Manage users",
            ),
            Permission(
                permission_id="roles:manage",
                action=PermissionAction.MANAGE_ROLES,
                resource_type=ResourceType.ROLE,
                description="Manage roles",
            ),
            Permission(
                permission_id="documents:manage",
                action=PermissionAction.MANAGE_DOCUMENTS,
                resource_type=ResourceType.DOCUMENT,
                description="Manage all documents",
            ),
            Permission(
                permission_id="analytics:view",
                action=PermissionAction.VIEW_ANALYTICS,
                resource_type=ResourceType.SYSTEM,
                description="View analytics",
            ),
            Permission(
                permission_id="audit:view",
                action=PermissionAction.VIEW_AUDIT_LOGS,
                resource_type=ResourceType.SYSTEM,
                description="View audit logs",
            ),
            Permission(
                permission_id="system:configure",
                action=PermissionAction.CONFIGURE_SYSTEM,
                resource_type=ResourceType.SYSTEM,
                description="Configure system settings",
            ),
            Permission(
                permission_id="api_keys:manage",
                action=PermissionAction.MANAGE_API_KEYS,
                resource_type=ResourceType.API_KEY,
                description="Manage API keys",
            ),
        ]

        for perm in permissions:
            await self.store.add_permission(perm)

    async def _create_default_roles(self) -> None:
        """Create default roles."""
        roles = [
            Role(
                role_id="admin",
                name="Administrator",
                description="Full system access",
                permissions=[
                    "document:read", "document:write", "document:delete", "document:share",
                    "query:execute", "query:with_context",
                    "users:manage", "roles:manage", "documents:manage",
                    "analytics:view", "audit:view", "system:configure", "api_keys:manage",
                ],
                is_system=True,
            ),
            Role(
                role_id="editor",
                name="Editor",
                description="Can read and write documents",
                permissions=[
                    "document:read", "document:write", "document:share",
                    "query:execute", "query:with_context",
                ],
                is_system=True,
            ),
            Role(
                role_id="viewer",
                name="Viewer",
                description="Read-only access",
                permissions=[
                    "document:read",
                    "query:execute",
                ],
                is_system=True,
            ),
            Role(
                role_id="api_user",
                name="API User",
                description="API access for integrations",
                permissions=[
                    "document:read",
                    "query:execute",
                ],
                is_system=True,
            ),
        ]

        for role in roles:
            await self.store.add_role(role)

    async def create_role(
        self,
        name: str,
        permissions: list[str],
        description: str = "",
        parent_roles: list[str] | None = None,
        **metadata,
    ) -> Role:
        """Create a new role."""
        role = Role(
            role_id=secrets.token_hex(8),
            name=name,
            description=description,
            permissions=permissions,
            parent_roles=parent_roles or [],
            metadata=metadata,
        )

        await self.store.add_role(role)

        logger.info(
            "Role created",
            role_id=role.role_id,
            name=name,
        )

        return role

    async def update_role(
        self,
        role_id: str,
        name: str | None = None,
        permissions: list[str] | None = None,
        description: str | None = None,
    ) -> Role | None:
        """Update a role."""
        role = await self.store.get_role(role_id)
        if not role:
            return None

        if role.is_system:
            logger.warning("Cannot modify system role", role_id=role_id)
            return None

        if name:
            role.name = name
        if permissions is not None:
            role.permissions = permissions
        if description is not None:
            role.description = description

        await self.store.update_role(role)

        return role

    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        return await self.store.delete_role(role_id)

    async def assign_role_to_user(self, user_id: str, role_id: str) -> None:
        """Assign a role to a user."""
        role = await self.store.get_role(role_id)
        if not role:
            raise ValueError(f"Role {role_id} not found")

        await self.store.assign_role(user_id, role_id)

        logger.info(
            "Role assigned",
            user_id=user_id,
            role_id=role_id,
        )

    async def revoke_role_from_user(self, user_id: str, role_id: str) -> None:
        """Revoke a role from a user."""
        await self.store.revoke_role(user_id, role_id)

        logger.info(
            "Role revoked",
            user_id=user_id,
            role_id=role_id,
        )

    async def get_user_permissions(self, user_id: str) -> list[Permission]:
        """Get all permissions for a user (including inherited)."""
        role_ids = await self.store.get_user_roles(user_id)
        all_permissions: set[str] = set()

        # Collect permissions from all roles
        for role_id in role_ids:
            role_permissions = await self._get_role_permissions_recursive(role_id)
            all_permissions.update(role_permissions)

        # Get actual permission objects
        permissions = []
        for perm_id in all_permissions:
            perm = await self.store.get_permission(perm_id)
            if perm:
                permissions.append(perm)

        return permissions

    async def _get_role_permissions_recursive(
        self,
        role_id: str,
        visited: set[str] | None = None,
    ) -> set[str]:
        """Get all permissions for a role including inherited."""
        visited = visited or set()
        if role_id in visited:
            return set()  # Prevent infinite loops

        visited.add(role_id)
        role = await self.store.get_role(role_id)
        if not role:
            return set()

        permissions = set(role.permissions)

        # Add permissions from parent roles
        for parent_id in role.parent_roles:
            parent_perms = await self._get_role_permissions_recursive(
                parent_id, visited
            )
            permissions.update(parent_perms)

        return permissions


class PermissionChecker:
    """Check user permissions for resources.

    Provides fast permission checks for authorization decisions.
    """

    def __init__(
        self,
        rbac_manager: RBACManager,
        settings: Settings | None = None,
    ):
        self.rbac = rbac_manager
        self.settings = settings or get_settings()
        # Cache for performance
        self._permission_cache: dict[str, list[Permission]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_times: dict[str, datetime] = {}

    async def check_permission(
        self,
        user_id: str,
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> bool:
        """Check if user has permission for an action.

        Args:
            user_id: User ID
            action: Action to check
            resource_type: Type of resource
            resource_id: Optional specific resource ID
            user_roles: Optional pre-fetched user roles

        Returns:
            True if user has permission
        """
        # Get user permissions (with caching)
        permissions = await self._get_cached_permissions(user_id)

        # Check for matching permission
        for perm in permissions:
            if perm.matches(action, resource_type, resource_id):
                return True

        # Check for admin override
        if user_roles is None:
            user_roles = await self.rbac.store.get_user_roles(user_id)

        if "admin" in user_roles:
            return True

        return False

    async def _get_cached_permissions(self, user_id: str) -> list[Permission]:
        """Get permissions with caching."""
        now = datetime.utcnow()

        # Check cache
        if user_id in self._permission_cache:
            cache_time = self._cache_times.get(user_id)
            if cache_time and (now - cache_time).total_seconds() < self._cache_ttl:
                return self._permission_cache[user_id]

        # Fetch and cache
        permissions = await self.rbac.get_user_permissions(user_id)
        self._permission_cache[user_id] = permissions
        self._cache_times[user_id] = now

        return permissions

    def invalidate_cache(self, user_id: str | None = None) -> None:
        """Invalidate permission cache."""
        if user_id:
            self._permission_cache.pop(user_id, None)
            self._cache_times.pop(user_id, None)
        else:
            self._permission_cache.clear()
            self._cache_times.clear()

    async def check_document_access(
        self,
        user_id: str,
        document_id: str,
        action: PermissionAction,
        user_roles: list[str] | None = None,
    ) -> bool:
        """Check if user has access to a specific document.

        Combines role-based permissions with document ACL.
        """
        # First check role-based permission
        has_permission = await self.check_permission(
            user_id,
            action,
            ResourceType.DOCUMENT,
            document_id,
            user_roles,
        )

        if has_permission:
            return True

        # Check document ACL
        acl = await self.rbac.store.get_document_acl(document_id)
        if not acl:
            # No ACL = use default permissions
            return has_permission

        if user_roles is None:
            user_roles = await self.rbac.store.get_user_roles(user_id)

        if action in (PermissionAction.READ, PermissionAction.QUERY):
            return acl.can_read(user_id, user_roles)
        elif action in (PermissionAction.WRITE, PermissionAction.DELETE):
            return acl.can_write(user_id, user_roles)
        elif action == PermissionAction.SHARE:
            return acl.can_write(user_id, user_roles)

        return False

    async def filter_accessible_documents(
        self,
        user_id: str,
        document_ids: list[str],
        user_roles: list[str] | None = None,
    ) -> list[str]:
        """Filter document IDs to only those accessible to user."""
        accessible = []

        for doc_id in document_ids:
            if await self.check_document_access(
                user_id,
                doc_id,
                PermissionAction.READ,
                user_roles,
            ):
                accessible.append(doc_id)

        return accessible

    async def get_acl_filter(
        self,
        user_id: str,
        user_roles: list[str] | None = None,
    ) -> list[str]:
        """Get ACL filter tags for vector search.

        Returns list of role/user tags to include in search filter.
        """
        if user_roles is None:
            user_roles = await self.rbac.store.get_user_roles(user_id)

        # Check if user is admin (can see everything)
        if "admin" in user_roles:
            return ["*"]

        # Build filter tags
        tags = [f"user:{user_id}"]
        for role in user_roles:
            tags.append(f"role:{role}")

        # Always include public
        tags.append("public")

        return tags


class DocumentACLManager:
    """Manage document access control lists."""

    def __init__(
        self,
        store: PermissionStore,
        settings: Settings | None = None,
    ):
        self.store = store
        self.settings = settings or get_settings()

    async def create_acl(
        self,
        document_id: str,
        owner_id: str,
        read_users: list[str] | None = None,
        write_users: list[str] | None = None,
        read_roles: list[str] | None = None,
        write_roles: list[str] | None = None,
        is_public: bool = False,
    ) -> DocumentACL:
        """Create ACL for a document."""
        acl = DocumentACL(
            document_id=document_id,
            owner_id=owner_id,
            read_users=read_users or [],
            write_users=write_users or [],
            read_roles=read_roles or [],
            write_roles=write_roles or [],
            is_public=is_public,
        )

        await self.store.set_document_acl(acl)

        logger.info(
            "Document ACL created",
            document_id=document_id,
            owner_id=owner_id,
        )

        return acl

    async def update_acl(
        self,
        document_id: str,
        read_users: list[str] | None = None,
        write_users: list[str] | None = None,
        read_roles: list[str] | None = None,
        write_roles: list[str] | None = None,
        is_public: bool | None = None,
    ) -> DocumentACL | None:
        """Update ACL for a document."""
        acl = await self.store.get_document_acl(document_id)
        if not acl:
            return None

        if read_users is not None:
            acl.read_users = read_users
        if write_users is not None:
            acl.write_users = write_users
        if read_roles is not None:
            acl.read_roles = read_roles
        if write_roles is not None:
            acl.write_roles = write_roles
        if is_public is not None:
            acl.is_public = is_public

        acl.updated_at = datetime.utcnow()
        await self.store.set_document_acl(acl)

        return acl

    async def share_document(
        self,
        document_id: str,
        user_ids: list[str] | None = None,
        role_ids: list[str] | None = None,
        write_access: bool = False,
    ) -> DocumentACL | None:
        """Share a document with users or roles."""
        acl = await self.store.get_document_acl(document_id)
        if not acl:
            return None

        if user_ids:
            if write_access:
                acl.write_users.extend(u for u in user_ids if u not in acl.write_users)
            else:
                acl.read_users.extend(u for u in user_ids if u not in acl.read_users)

        if role_ids:
            if write_access:
                acl.write_roles.extend(r for r in role_ids if r not in acl.write_roles)
            else:
                acl.read_roles.extend(r for r in role_ids if r not in acl.read_roles)

        acl.updated_at = datetime.utcnow()
        await self.store.set_document_acl(acl)

        logger.info(
            "Document shared",
            document_id=document_id,
            user_ids=user_ids,
            role_ids=role_ids,
        )

        return acl

    async def unshare_document(
        self,
        document_id: str,
        user_ids: list[str] | None = None,
        role_ids: list[str] | None = None,
    ) -> DocumentACL | None:
        """Remove sharing from a document."""
        acl = await self.store.get_document_acl(document_id)
        if not acl:
            return None

        if user_ids:
            acl.read_users = [u for u in acl.read_users if u not in user_ids]
            acl.write_users = [u for u in acl.write_users if u not in user_ids]

        if role_ids:
            acl.read_roles = [r for r in acl.read_roles if r not in role_ids]
            acl.write_roles = [r for r in acl.write_roles if r not in role_ids]

        acl.updated_at = datetime.utcnow()
        await self.store.set_document_acl(acl)

        return acl

    async def get_acl_tags(self, acl: DocumentACL) -> list[str]:
        """Get ACL tags for indexing in vector store."""
        tags = []

        # Owner always has access
        tags.append(f"user:{acl.owner_id}")

        # Read users
        for user_id in acl.read_users:
            tags.append(f"user:{user_id}")

        # Write users
        for user_id in acl.write_users:
            tags.append(f"user:{user_id}")

        # Read roles
        for role_id in acl.read_roles:
            tags.append(f"role:{role_id}")

        # Write roles
        for role_id in acl.write_roles:
            tags.append(f"role:{role_id}")

        # Public access
        if acl.is_public:
            tags.append("public")

        return list(set(tags))  # Deduplicate


# Module-level singletons
_rbac_manager: RBACManager | None = None
_permission_checker: PermissionChecker | None = None


async def get_rbac_manager(
    surrealdb_client: Any = None,
) -> RBACManager:
    """Get or create RBAC manager."""
    global _rbac_manager
    if _rbac_manager is None:
        store = PermissionStore(surrealdb_client)
        _rbac_manager = RBACManager(store)
        await _rbac_manager.initialize()
    return _rbac_manager


async def get_permission_checker() -> PermissionChecker:
    """Get or create permission checker."""
    global _permission_checker
    if _permission_checker is None:
        rbac = await get_rbac_manager()
        _permission_checker = PermissionChecker(rbac)
    return _permission_checker
