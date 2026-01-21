"""GDPR compliance module.

Provides data subject rights management, consent tracking,
data retention policies, anonymization, and right to erasure.
"""

import asyncio
import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class ConsentType(str, Enum):
    """Types of consent."""

    DATA_PROCESSING = "data_processing"
    PERSONALIZATION = "personalization"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    THIRD_PARTY_SHARING = "third_party_sharing"
    COOKIES = "cookies"
    DATA_RETENTION = "data_retention"


class ConsentStatus(str, Enum):
    """Status of consent."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class DataCategory(str, Enum):
    """Categories of personal data."""

    IDENTITY = "identity"  # Name, email, etc.
    CONTACT = "contact"  # Address, phone
    USAGE = "usage"  # Usage data, queries
    PREFERENCES = "preferences"  # User preferences
    BEHAVIORAL = "behavioral"  # Behavioral data
    CONTENT = "content"  # User-generated content
    TECHNICAL = "technical"  # IP, device info
    SENSITIVE = "sensitive"  # Special category data


class RequestType(str, Enum):
    """Types of data subject requests."""

    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure
    RESTRICTION = "restriction"  # Right to restriction
    PORTABILITY = "portability"  # Right to portability
    OBJECTION = "objection"  # Right to object


class RequestStatus(str, Enum):
    """Status of a data subject request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Consent:
    """User consent record."""

    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: datetime | None = None
    withdrawn_at: datetime | None = None
    expires_at: datetime | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "consent_type": self.consent_type.value,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "is_valid": self.is_valid(),
        }


@dataclass
class DataSubjectRequest:
    """A data subject request."""

    request_id: str
    user_id: str
    request_type: RequestType
    status: RequestStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    due_date: datetime | None = None
    data_categories: list[DataCategory] = field(default_factory=list)
    notes: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "data_categories": [c.value for c in self.data_categories],
            "notes": self.notes,
        }


@dataclass
class RetentionPolicy:
    """Data retention policy."""

    policy_id: str
    data_category: DataCategory
    retention_days: int
    description: str = ""
    legal_basis: str = ""
    exceptions: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "data_category": self.data_category.value,
            "retention_days": self.retention_days,
            "description": self.description,
            "legal_basis": self.legal_basis,
            "enabled": self.enabled,
        }


class ConsentManager:
    """Manage user consent.

    Tracks and manages consent for various processing activities.
    """

    def __init__(
        self,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client
        self._consents: dict[str, dict[ConsentType, Consent]] = {}
        self._lock = asyncio.Lock()

    async def grant_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        ip_address: str | None = None,
        user_agent: str | None = None,
        duration_days: int | None = None,
        version: str = "1.0",
        **metadata,
    ) -> Consent:
        """Grant consent for a processing activity."""
        now = datetime.utcnow()
        expires_at = None
        if duration_days:
            expires_at = now + timedelta(days=duration_days)

        consent = Consent(
            user_id=user_id,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            version=version,
            metadata=metadata,
        )

        async with self._lock:
            if user_id not in self._consents:
                self._consents[user_id] = {}
            self._consents[user_id][consent_type] = consent

        # Persist to database
        if self.surrealdb:
            await self._store_consent(consent)

        logger.info(
            "Consent granted",
            user_id=user_id,
            consent_type=consent_type.value,
        )

        return consent

    async def withdraw_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> Consent | None:
        """Withdraw consent for a processing activity."""
        async with self._lock:
            if user_id not in self._consents:
                return None
            if consent_type not in self._consents[user_id]:
                return None

            consent = self._consents[user_id][consent_type]
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.utcnow()

        # Update in database
        if self.surrealdb:
            await self._store_consent(consent)

        logger.info(
            "Consent withdrawn",
            user_id=user_id,
            consent_type=consent_type.value,
        )

        return consent

    async def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """Check if user has valid consent for a processing activity."""
        async with self._lock:
            if user_id not in self._consents:
                # Try loading from database
                consent = await self._load_consent(user_id, consent_type)
                if consent:
                    if user_id not in self._consents:
                        self._consents[user_id] = {}
                    self._consents[user_id][consent_type] = consent
                else:
                    return False

            if consent_type not in self._consents.get(user_id, {}):
                return False

            return self._consents[user_id][consent_type].is_valid()

    async def get_user_consents(self, user_id: str) -> list[Consent]:
        """Get all consents for a user."""
        async with self._lock:
            consents = list(self._consents.get(user_id, {}).values())

        # Also load from database
        if self.surrealdb:
            db_consents = await self._load_all_consents(user_id)
            # Merge with in-memory (in-memory takes precedence)
            seen_types = {c.consent_type for c in consents}
            for c in db_consents:
                if c.consent_type not in seen_types:
                    consents.append(c)

        return consents

    async def _store_consent(self, consent: Consent) -> None:
        """Store consent in database."""
        if not self.surrealdb:
            return

        try:
            await self.surrealdb.client.query(
                """
                UPSERT consent SET
                    user_id = $user_id,
                    consent_type = $consent_type,
                    status = $status,
                    granted_at = $granted_at,
                    withdrawn_at = $withdrawn_at,
                    expires_at = $expires_at,
                    ip_address = $ip_address,
                    version = $version,
                    metadata = $metadata
                WHERE user_id = $user_id AND consent_type = $consent_type
                """,
                {
                    "user_id": consent.user_id,
                    "consent_type": consent.consent_type.value,
                    "status": consent.status.value,
                    "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
                    "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
                    "ip_address": consent.ip_address,
                    "version": consent.version,
                    "metadata": consent.metadata,
                },
            )
        except Exception as e:
            logger.error("Failed to store consent", error=str(e))

    async def _load_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> Consent | None:
        """Load consent from database."""
        if not self.surrealdb:
            return None

        try:
            result = await self.surrealdb.client.query(
                "SELECT * FROM consent WHERE user_id = $user_id AND consent_type = $consent_type",
                {"user_id": user_id, "consent_type": consent_type.value},
            )
            if result and result[0].get("result"):
                data = result[0]["result"][0]
                return self._parse_consent(data)
        except Exception as e:
            logger.error("Failed to load consent", error=str(e))

        return None

    async def _load_all_consents(self, user_id: str) -> list[Consent]:
        """Load all consents for a user from database."""
        if not self.surrealdb:
            return []

        try:
            result = await self.surrealdb.client.query(
                "SELECT * FROM consent WHERE user_id = $user_id",
                {"user_id": user_id},
            )
            if result and result[0].get("result"):
                return [self._parse_consent(d) for d in result[0]["result"]]
        except Exception as e:
            logger.error("Failed to load consents", error=str(e))

        return []

    def _parse_consent(self, data: dict) -> Consent:
        """Parse consent from database record."""
        return Consent(
            user_id=data["user_id"],
            consent_type=ConsentType(data["consent_type"]),
            status=ConsentStatus(data["status"]),
            granted_at=datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None,
            withdrawn_at=datetime.fromisoformat(data["withdrawn_at"]) if data.get("withdrawn_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            ip_address=data.get("ip_address"),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )


class DataRetentionPolicy:
    """Manage data retention policies.

    Handles automatic data cleanup based on retention policies.
    """

    def __init__(
        self,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client
        self._policies: dict[str, RetentionPolicy] = {}
        self._lock = asyncio.Lock()

        # Initialize default policies
        self._init_default_policies()

    def _init_default_policies(self) -> None:
        """Initialize default retention policies."""
        defaults = [
            RetentionPolicy(
                policy_id="usage_data",
                data_category=DataCategory.USAGE,
                retention_days=90,
                description="User query and usage data",
                legal_basis="Legitimate interest for service improvement",
            ),
            RetentionPolicy(
                policy_id="behavioral_data",
                data_category=DataCategory.BEHAVIORAL,
                retention_days=30,
                description="User interaction and behavioral data",
                legal_basis="Consent for personalization",
            ),
            RetentionPolicy(
                policy_id="technical_data",
                data_category=DataCategory.TECHNICAL,
                retention_days=14,
                description="Technical logs and IP addresses",
                legal_basis="Security and fraud prevention",
            ),
            RetentionPolicy(
                policy_id="content_data",
                data_category=DataCategory.CONTENT,
                retention_days=365,
                description="User-uploaded content",
                legal_basis="Contract performance",
            ),
        ]

        for policy in defaults:
            self._policies[policy.policy_id] = policy

    async def add_policy(self, policy: RetentionPolicy) -> None:
        """Add or update a retention policy."""
        async with self._lock:
            self._policies[policy.policy_id] = policy

        logger.info(
            "Retention policy added",
            policy_id=policy.policy_id,
            retention_days=policy.retention_days,
        )

    async def get_policy(self, policy_id: str) -> RetentionPolicy | None:
        """Get a retention policy."""
        return self._policies.get(policy_id)

    async def get_policies(self) -> list[RetentionPolicy]:
        """Get all retention policies."""
        return list(self._policies.values())

    async def apply_retention(
        self,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Apply retention policies and clean up expired data.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Summary of deleted/deletable data
        """
        if not self.surrealdb:
            return {"error": "Database not configured"}

        results = {}
        now = datetime.utcnow()

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            cutoff_date = now - timedelta(days=policy.retention_days)

            # Get data tables for this category
            tables = self._get_tables_for_category(policy.data_category)

            for table in tables:
                try:
                    if dry_run:
                        # Count records that would be deleted
                        result = await self.surrealdb.client.query(
                            f"SELECT count() FROM {table} WHERE created_at < $cutoff",
                            {"cutoff": cutoff_date.isoformat()},
                        )
                        count = result[0]["result"][0]["count"] if result and result[0].get("result") else 0
                        results[f"{table}_{policy.policy_id}"] = {
                            "would_delete": count,
                            "cutoff_date": cutoff_date.isoformat(),
                        }
                    else:
                        # Actually delete the records
                        result = await self.surrealdb.client.query(
                            f"DELETE FROM {table} WHERE created_at < $cutoff",
                            {"cutoff": cutoff_date.isoformat()},
                        )
                        results[f"{table}_{policy.policy_id}"] = {
                            "deleted": True,
                            "cutoff_date": cutoff_date.isoformat(),
                        }
                        logger.info(
                            "Applied retention policy",
                            table=table,
                            policy_id=policy.policy_id,
                            cutoff_date=cutoff_date.isoformat(),
                        )

                except Exception as e:
                    logger.error(
                        "Failed to apply retention policy",
                        table=table,
                        error=str(e),
                    )
                    results[f"{table}_{policy.policy_id}"] = {"error": str(e)}

        return results

    def _get_tables_for_category(self, category: DataCategory) -> list[str]:
        """Get database tables for a data category."""
        mapping = {
            DataCategory.USAGE: ["query_log", "feedback"],
            DataCategory.BEHAVIORAL: ["user_interaction", "click_log"],
            DataCategory.TECHNICAL: ["access_log", "error_log"],
            DataCategory.CONTENT: ["user_document"],
            DataCategory.PREFERENCES: ["user_profile", "user_preference"],
        }
        return mapping.get(category, [])


class AnonymizationService:
    """Anonymize PII in data.

    Provides various anonymization techniques for personal data.
    """

    # Common PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "credit_card": r"\b\d{4}[-]?\d{4}[-]?\d{4}[-]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "name": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
    }

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PII_PATTERNS.items()
        }

    async def anonymize_text(
        self,
        text: str,
        pii_types: list[str] | None = None,
        replacement: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Anonymize PII in text.

        Args:
            text: Text to anonymize
            pii_types: Types of PII to anonymize (None = all)
            replacement: Replacement string (None = type-specific)

        Returns:
            Tuple of (anonymized text, counts by PII type)
        """
        if pii_types is None:
            pii_types = list(self.PII_PATTERNS.keys())

        counts = {}
        result = text

        for pii_type in pii_types:
            if pii_type not in self._compiled_patterns:
                continue

            pattern = self._compiled_patterns[pii_type]
            matches = pattern.findall(result)
            counts[pii_type] = len(matches)

            if replacement:
                result = pattern.sub(replacement, result)
            else:
                # Type-specific replacement
                repl = f"[{pii_type.upper()}_REDACTED]"
                result = pattern.sub(repl, result)

        return result, counts

    async def anonymize_dict(
        self,
        data: dict[str, Any],
        fields_to_anonymize: list[str] | None = None,
        pii_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Anonymize PII in a dictionary.

        Args:
            data: Dictionary to anonymize
            fields_to_anonymize: Specific fields to anonymize (None = all string fields)
            pii_types: Types of PII to anonymize

        Returns:
            Anonymized dictionary
        """
        result = {}

        for key, value in data.items():
            if fields_to_anonymize and key not in fields_to_anonymize:
                result[key] = value
            elif isinstance(value, str):
                anonymized, _ = await self.anonymize_text(value, pii_types)
                result[key] = anonymized
            elif isinstance(value, dict):
                result[key] = await self.anonymize_dict(value, fields_to_anonymize, pii_types)
            elif isinstance(value, list):
                result[key] = [
                    await self.anonymize_dict(item, fields_to_anonymize, pii_types)
                    if isinstance(item, dict)
                    else (await self.anonymize_text(item, pii_types))[0]
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    async def hash_identifier(
        self,
        identifier: str,
        salt: str | None = None,
    ) -> str:
        """Hash an identifier for pseudonymization."""
        salt = salt or secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
        return hashed[:16]

    async def generalize_location(
        self,
        latitude: float,
        longitude: float,
        precision: int = 2,
    ) -> tuple[float, float]:
        """Generalize location by reducing precision."""
        return (
            round(latitude, precision),
            round(longitude, precision),
        )

    async def generalize_age(self, age: int, bucket_size: int = 10) -> str:
        """Generalize age into buckets."""
        lower = (age // bucket_size) * bucket_size
        upper = lower + bucket_size - 1
        return f"{lower}-{upper}"


class RightToErasure:
    """Handle right to erasure (right to be forgotten) requests.

    Provides complete deletion of all user data across all systems.
    """

    def __init__(
        self,
        surrealdb_client: Any = None,
        qdrant_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client
        self.qdrant = qdrant_client
        self._deletion_hooks: list[Callable[[str], Any]] = []

    def register_deletion_hook(self, hook: Callable[[str], Any]) -> None:
        """Register a hook to be called during user deletion."""
        self._deletion_hooks.append(hook)

    async def erase_user_data(
        self,
        user_id: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Erase all data for a user.

        Args:
            user_id: User ID to erase
            dry_run: If True, only report what would be deleted

        Returns:
            Summary of deleted data
        """
        results = {
            "user_id": user_id,
            "dry_run": dry_run,
            "deleted": {},
            "errors": [],
        }

        # List of tables containing user data
        user_tables = [
            "user_profile",
            "user_preference",
            "consent",
            "query_log",
            "feedback",
            "user_interaction",
            "user_document",
            "api_key",
            "session",
            "data_subject_request",
        ]

        # Delete from SurrealDB tables
        if self.surrealdb:
            for table in user_tables:
                try:
                    if dry_run:
                        result = await self.surrealdb.client.query(
                            f"SELECT count() FROM {table} WHERE user_id = $user_id",
                            {"user_id": user_id},
                        )
                        count = result[0]["result"][0]["count"] if result and result[0].get("result") else 0
                        results["deleted"][table] = {"would_delete": count}
                    else:
                        await self.surrealdb.client.query(
                            f"DELETE FROM {table} WHERE user_id = $user_id",
                            {"user_id": user_id},
                        )
                        results["deleted"][table] = {"deleted": True}
                except Exception as e:
                    results["errors"].append(f"{table}: {str(e)}")

        # Delete from vector database
        if self.qdrant:
            try:
                if not dry_run:
                    # Delete vectors with user_id in metadata
                    await self.qdrant.delete_by_filter(
                        collection="chunks",
                        filter_conditions={"user_id": user_id},
                    )
                results["deleted"]["qdrant_vectors"] = {"deleted": not dry_run}
            except Exception as e:
                results["errors"].append(f"qdrant: {str(e)}")

        # Call deletion hooks
        if not dry_run:
            for hook in self._deletion_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(user_id)
                    else:
                        hook(user_id)
                except Exception as e:
                    results["errors"].append(f"hook: {str(e)}")

        if not dry_run:
            logger.info(
                "User data erased",
                user_id=user_id,
                tables_deleted=len(results["deleted"]),
                errors=len(results["errors"]),
            )

        return results


class DataSubjectManager:
    """Manage data subject requests.

    Handles access, rectification, erasure, and other GDPR rights requests.
    """

    # GDPR deadline is 30 days
    DEFAULT_DEADLINE_DAYS = 30

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        retention_policy: DataRetentionPolicy | None = None,
        anonymization_service: AnonymizationService | None = None,
        right_to_erasure: RightToErasure | None = None,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client
        self.consent = consent_manager or ConsentManager(surrealdb_client, settings)
        self.retention = retention_policy or DataRetentionPolicy(surrealdb_client, settings)
        self.anonymization = anonymization_service or AnonymizationService(settings)
        self.erasure = right_to_erasure or RightToErasure(surrealdb_client, settings=settings)
        self._requests: dict[str, DataSubjectRequest] = {}
        self._lock = asyncio.Lock()

    async def create_request(
        self,
        user_id: str,
        request_type: RequestType,
        data_categories: list[DataCategory] | None = None,
        notes: str = "",
        **metadata,
    ) -> DataSubjectRequest:
        """Create a new data subject request."""
        now = datetime.utcnow()
        request = DataSubjectRequest(
            request_id=secrets.token_hex(8),
            user_id=user_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            created_at=now,
            updated_at=now,
            due_date=now + timedelta(days=self.DEFAULT_DEADLINE_DAYS),
            data_categories=data_categories or [],
            notes=notes,
            metadata=metadata,
        )

        async with self._lock:
            self._requests[request.request_id] = request

        # Persist to database
        if self.surrealdb:
            await self._store_request(request)

        logger.info(
            "Data subject request created",
            request_id=request.request_id,
            request_type=request_type.value,
            user_id=user_id,
        )

        return request

    async def process_request(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        """Process a data subject request.

        Args:
            request_id: Request ID to process

        Returns:
            Result of processing
        """
        request = await self.get_request(request_id)
        if not request:
            return {"error": "Request not found"}

        # Update status
        request.status = RequestStatus.IN_PROGRESS
        request.updated_at = datetime.utcnow()

        result = {}

        try:
            if request.request_type == RequestType.ACCESS:
                result = await self._handle_access_request(request)
            elif request.request_type == RequestType.ERASURE:
                result = await self._handle_erasure_request(request)
            elif request.request_type == RequestType.PORTABILITY:
                result = await self._handle_portability_request(request)
            elif request.request_type == RequestType.RECTIFICATION:
                result = await self._handle_rectification_request(request)
            elif request.request_type == RequestType.RESTRICTION:
                result = await self._handle_restriction_request(request)
            elif request.request_type == RequestType.OBJECTION:
                result = await self._handle_objection_request(request)

            request.status = RequestStatus.COMPLETED
            request.completed_at = datetime.utcnow()
            request.result = result

        except Exception as e:
            logger.error("Failed to process request", request_id=request_id, error=str(e))
            request.status = RequestStatus.PENDING
            result = {"error": str(e)}

        request.updated_at = datetime.utcnow()
        await self._store_request(request)

        return result

    async def _handle_access_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle right to access request."""
        user_id = request.user_id
        data = {}

        if not self.surrealdb:
            return {"error": "Database not configured"}

        # Collect data from various tables
        tables = [
            "user_profile",
            "query_log",
            "feedback",
            "consent",
            "user_interaction",
        ]

        for table in tables:
            try:
                result = await self.surrealdb.client.query(
                    f"SELECT * FROM {table} WHERE user_id = $user_id",
                    {"user_id": user_id},
                )
                if result and result[0].get("result"):
                    data[table] = result[0]["result"]
            except Exception as e:
                logger.warning(f"Failed to fetch {table}", error=str(e))

        return {
            "user_id": user_id,
            "data": data,
            "exported_at": datetime.utcnow().isoformat(),
        }

    async def _handle_erasure_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle right to erasure request."""
        return await self.erasure.erase_user_data(request.user_id)

    async def _handle_portability_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle data portability request."""
        # Similar to access but in portable format (JSON)
        access_result = await self._handle_access_request(request)
        return {
            "format": "JSON",
            "data": json.dumps(access_result, default=str, indent=2),
            "exported_at": datetime.utcnow().isoformat(),
        }

    async def _handle_rectification_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle rectification request."""
        # Rectification requires manual intervention
        return {
            "status": "requires_manual_review",
            "notes": request.notes,
        }

    async def _handle_restriction_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle restriction of processing request."""
        # Withdraw all consents
        consents = await self.consent.get_user_consents(request.user_id)
        withdrawn = []

        for consent in consents:
            if consent.is_valid():
                await self.consent.withdraw_consent(
                    request.user_id,
                    consent.consent_type,
                )
                withdrawn.append(consent.consent_type.value)

        return {
            "consents_withdrawn": withdrawn,
        }

    async def _handle_objection_request(
        self,
        request: DataSubjectRequest,
    ) -> dict[str, Any]:
        """Handle objection to processing request."""
        # Similar to restriction
        return await self._handle_restriction_request(request)

    async def get_request(self, request_id: str) -> DataSubjectRequest | None:
        """Get a data subject request."""
        async with self._lock:
            return self._requests.get(request_id)

    async def list_requests(
        self,
        user_id: str | None = None,
        status: RequestStatus | None = None,
    ) -> list[DataSubjectRequest]:
        """List data subject requests."""
        async with self._lock:
            requests = list(self._requests.values())

        if user_id:
            requests = [r for r in requests if r.user_id == user_id]
        if status:
            requests = [r for r in requests if r.status == status]

        return requests

    async def _store_request(self, request: DataSubjectRequest) -> None:
        """Store request in database."""
        if not self.surrealdb:
            return

        try:
            await self.surrealdb.client.query(
                """
                UPSERT data_subject_request SET
                    request_id = $request_id,
                    user_id = $user_id,
                    request_type = $request_type,
                    status = $status,
                    created_at = $created_at,
                    updated_at = $updated_at,
                    completed_at = $completed_at,
                    due_date = $due_date,
                    data_categories = $data_categories,
                    notes = $notes,
                    result = $result,
                    metadata = $metadata
                WHERE request_id = $request_id
                """,
                {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "request_type": request.request_type.value,
                    "status": request.status.value,
                    "created_at": request.created_at.isoformat(),
                    "updated_at": request.updated_at.isoformat(),
                    "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                    "due_date": request.due_date.isoformat() if request.due_date else None,
                    "data_categories": [c.value for c in request.data_categories],
                    "notes": request.notes,
                    "result": request.result,
                    "metadata": request.metadata,
                },
            )
        except Exception as e:
            logger.error("Failed to store request", error=str(e))


# Module-level singletons
_data_subject_manager: DataSubjectManager | None = None


async def get_data_subject_manager(
    surrealdb_client: Any = None,
) -> DataSubjectManager:
    """Get or create data subject manager."""
    global _data_subject_manager
    if _data_subject_manager is None:
        _data_subject_manager = DataSubjectManager(surrealdb_client=surrealdb_client)
    return _data_subject_manager
