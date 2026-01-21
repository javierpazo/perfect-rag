"""Audit logging for compliance and security.

Provides structured audit logging, storage, querying, and
compliance reporting for RAG systems.
"""

import asyncio
import gzip
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # Data events
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_ACCESSED = "document_accessed"
    DOCUMENT_SHARED = "document_shared"
    QUERY_EXECUTED = "query_executed"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"

    # GDPR events
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_SUBJECT_REQUEST = "data_subject_request"
    DATA_ERASURE = "data_erasure"
    DATA_PORTABILITY = "data_portability"

    # Admin events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    CONFIG_CHANGED = "config_changed"
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"

    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A structured audit log entry."""

    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO

    # Actor information
    user_id: str | None = None
    username: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    session_id: str | None = None

    # Action details
    action: str = ""
    resource_type: str | None = None
    resource_id: str | None = None
    description: str = ""

    # Result
    success: bool = True
    error_message: str | None = None

    # Context
    request_id: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Compliance
    contains_pii: bool = False
    data_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "description": self.description,
            "success": self.success,
            "error_message": self.error_message,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
            "contains_pii": self.contains_pii,
            "data_categories": self.data_categories,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data.get("severity", "info")),
            user_id=data.get("user_id"),
            username=data.get("username"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            session_id=data.get("session_id"),
            action=data.get("action", ""),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            description=data.get("description", ""),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            request_id=data.get("request_id"),
            trace_id=data.get("trace_id"),
            metadata=data.get("metadata", {}),
            contains_pii=data.get("contains_pii", False),
            data_categories=data.get("data_categories", []),
        )


class AuditStorage:
    """Store audit logs to file and database.

    Provides:
    - File-based logging with rotation
    - Database persistence
    - Compression for archived logs
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client

        # File storage
        self.log_dir = Path(log_dir) if log_dir else Path("./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_file: TextIO | None = None
        self._current_file_date: datetime | None = None
        self._lock = asyncio.Lock()

        # In-memory buffer for batch writes
        self._buffer: list[AuditEntry] = []
        self._buffer_size = 100
        self._last_flush = datetime.utcnow()

    async def write(self, entry: AuditEntry) -> None:
        """Write an audit entry to storage."""
        async with self._lock:
            # Add to buffer
            self._buffer.append(entry)

            # Flush if buffer is full or time elapsed
            should_flush = (
                len(self._buffer) >= self._buffer_size
                or (datetime.utcnow() - self._last_flush).total_seconds() > 10
            )

            if should_flush:
                await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to storage."""
        if not self._buffer:
            return

        entries_to_write = self._buffer[:]
        self._buffer.clear()
        self._last_flush = datetime.utcnow()

        # Write to file
        await self._write_to_file(entries_to_write)

        # Write to database
        if self.surrealdb:
            await self._write_to_db(entries_to_write)

    async def _write_to_file(self, entries: list[AuditEntry]) -> None:
        """Write entries to file."""
        today = datetime.utcnow().date()

        # Check if we need a new file
        if (
            self._current_file is None
            or self._current_file_date != today
        ):
            if self._current_file:
                self._current_file.close()
                # Compress old log file
                await self._compress_old_log(self._current_file_date)

            # Create new log file
            filename = f"audit_{today.isoformat()}.jsonl"
            filepath = self.log_dir / filename
            self._current_file = open(filepath, "a", encoding="utf-8")
            self._current_file_date = today

        # Write entries
        for entry in entries:
            self._current_file.write(entry.to_json() + "\n")

        self._current_file.flush()

    async def _write_to_db(self, entries: list[AuditEntry]) -> None:
        """Write entries to database."""
        if not self.surrealdb:
            return

        for entry in entries:
            try:
                await self.surrealdb.client.query(
                    "CREATE audit_log CONTENT $data",
                    {"data": entry.to_dict()},
                )
            except Exception as e:
                logger.error("Failed to write audit entry to DB", error=str(e))

    async def _compress_old_log(self, date: datetime | None) -> None:
        """Compress old log file."""
        if not date:
            return

        filename = f"audit_{date.isoformat()}.jsonl"
        filepath = self.log_dir / filename

        if filepath.exists() and not filepath.with_suffix(".jsonl.gz").exists():
            try:
                with open(filepath, "rb") as f_in:
                    with gzip.open(filepath.with_suffix(".jsonl.gz"), "wb") as f_out:
                        f_out.writelines(f_in)
                # Remove original after successful compression
                filepath.unlink()
            except Exception as e:
                logger.error("Failed to compress audit log", error=str(e))

    async def read_entries(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEntry]:
        """Read audit entries from storage."""
        entries = []

        # Try database first (more efficient queries)
        if self.surrealdb:
            entries = await self._read_from_db(
                start_date, end_date, event_types, user_id, limit
            )
        else:
            # Fall back to file reading
            entries = await self._read_from_files(
                start_date, end_date, event_types, user_id, limit
            )

        return entries

    async def _read_from_db(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        event_types: list[AuditEventType] | None,
        user_id: str | None,
        limit: int,
    ) -> list[AuditEntry]:
        """Read entries from database."""
        if not self.surrealdb:
            return []

        conditions = []
        params = {"limit": limit}

        if start_date:
            conditions.append("timestamp >= $start_date")
            params["start_date"] = start_date.isoformat()

        if end_date:
            conditions.append("timestamp <= $end_date")
            params["end_date"] = end_date.isoformat()

        if event_types:
            conditions.append("event_type IN $event_types")
            params["event_types"] = [e.value for e in event_types]

        if user_id:
            conditions.append("user_id = $user_id")
            params["user_id"] = user_id

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM audit_log
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT $limit
        """

        try:
            result = await self.surrealdb.client.query(query, params)
            if result and result[0].get("result"):
                return [AuditEntry.from_dict(r) for r in result[0]["result"]]
        except Exception as e:
            logger.error("Failed to read audit entries from DB", error=str(e))

        return []

    async def _read_from_files(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        event_types: list[AuditEventType] | None,
        user_id: str | None,
        limit: int,
    ) -> list[AuditEntry]:
        """Read entries from files."""
        entries = []

        # Get relevant log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl*"), reverse=True)

        for log_file in log_files:
            if len(entries) >= limit:
                break

            try:
                # Handle compressed files
                if log_file.suffix == ".gz":
                    with gzip.open(log_file, "rt", encoding="utf-8") as f:
                        lines = f.readlines()
                else:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                for line in lines:
                    if len(entries) >= limit:
                        break

                    try:
                        data = json.loads(line)
                        entry = AuditEntry.from_dict(data)

                        # Apply filters
                        if start_date and entry.timestamp < start_date:
                            continue
                        if end_date and entry.timestamp > end_date:
                            continue
                        if event_types and entry.event_type not in event_types:
                            continue
                        if user_id and entry.user_id != user_id:
                            continue

                        entries.append(entry)

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Failed to parse audit entry", error=str(e))

            except Exception as e:
                logger.error("Failed to read audit file", file=str(log_file), error=str(e))

        return entries

    async def close(self) -> None:
        """Close storage and flush remaining entries."""
        async with self._lock:
            await self._flush()
            if self._current_file:
                self._current_file.close()
                self._current_file = None


class AuditLogger:
    """Log security-relevant events for compliance.

    Provides structured logging with automatic enrichment
    and compliance-aware features.
    """

    def __init__(
        self,
        storage: AuditStorage | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.storage = storage or AuditStorage(settings=settings)
        self._context: dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set context to be included in all log entries."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear context."""
        self._context.clear()

    async def log(
        self,
        event_type: AuditEventType,
        action: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: str | None = None,
        username: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        description: str = "",
        success: bool = True,
        error_message: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        contains_pii: bool = False,
        data_categories: list[str] | None = None,
        **metadata,
    ) -> AuditEntry:
        """Log an audit event.

        Args:
            event_type: Type of event
            action: Action performed
            severity: Severity level
            user_id: User who performed the action
            username: Username
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            description: Human-readable description
            success: Whether the action succeeded
            error_message: Error message if failed
            request_id: Request identifier
            trace_id: Trace identifier for distributed tracing
            contains_pii: Whether the entry contains PII
            data_categories: Categories of data involved
            **metadata: Additional metadata

        Returns:
            The created audit entry
        """
        entry = AuditEntry(
            entry_id=secrets.token_hex(8),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            user_id=user_id or self._context.get("user_id"),
            username=username or self._context.get("username"),
            ip_address=ip_address or self._context.get("ip_address"),
            user_agent=user_agent or self._context.get("user_agent"),
            session_id=session_id or self._context.get("session_id"),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            success=success,
            error_message=error_message,
            request_id=request_id or self._context.get("request_id"),
            trace_id=trace_id or self._context.get("trace_id"),
            metadata={**self._context.get("metadata", {}), **metadata},
            contains_pii=contains_pii,
            data_categories=data_categories or [],
        )

        await self.storage.write(entry)

        # Also log to structlog for real-time monitoring
        log_method = getattr(logger, severity.value, logger.info)
        log_method(
            "Audit event",
            event_type=event_type.value,
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
        )

        return entry

    # Convenience methods for common events

    async def log_login_success(
        self,
        user_id: str,
        username: str,
        ip_address: str | None = None,
        **kwargs,
    ) -> AuditEntry:
        """Log successful login."""
        return await self.log(
            AuditEventType.LOGIN_SUCCESS,
            action="login",
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            description=f"User {username} logged in successfully",
            **kwargs,
        )

    async def log_login_failure(
        self,
        username: str,
        ip_address: str | None = None,
        reason: str = "Invalid credentials",
        **kwargs,
    ) -> AuditEntry:
        """Log failed login attempt."""
        return await self.log(
            AuditEventType.LOGIN_FAILURE,
            action="login",
            severity=AuditSeverity.WARNING,
            username=username,
            ip_address=ip_address,
            description=f"Failed login attempt for {username}",
            success=False,
            error_message=reason,
            **kwargs,
        )

    async def log_access_denied(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        **kwargs,
    ) -> AuditEntry:
        """Log access denied event."""
        return await self.log(
            AuditEventType.ACCESS_DENIED,
            action=action,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            description=f"Access denied to {resource_type}:{resource_id}",
            success=False,
            **kwargs,
        )

    async def log_document_access(
        self,
        user_id: str,
        document_id: str,
        action: str = "read",
        **kwargs,
    ) -> AuditEntry:
        """Log document access."""
        return await self.log(
            AuditEventType.DOCUMENT_ACCESSED,
            action=action,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            description=f"Document {document_id} accessed",
            **kwargs,
        )

    async def log_query(
        self,
        user_id: str,
        query: str,
        chunks_retrieved: int = 0,
        **kwargs,
    ) -> AuditEntry:
        """Log query execution."""
        return await self.log(
            AuditEventType.QUERY_EXECUTED,
            action="query",
            user_id=user_id,
            resource_type="query",
            description=f"Query executed: {query[:100]}...",
            metadata={"query_preview": query[:200], "chunks_retrieved": chunks_retrieved},
            **kwargs,
        )

    async def log_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        request_id: str,
        **kwargs,
    ) -> AuditEntry:
        """Log GDPR data subject request."""
        return await self.log(
            AuditEventType.DATA_SUBJECT_REQUEST,
            action=request_type,
            user_id=user_id,
            resource_type="data_subject_request",
            resource_id=request_id,
            description=f"Data subject request: {request_type}",
            contains_pii=True,
            **kwargs,
        )

    async def log_data_erasure(
        self,
        user_id: str,
        tables_deleted: list[str],
        **kwargs,
    ) -> AuditEntry:
        """Log data erasure (right to be forgotten)."""
        return await self.log(
            AuditEventType.DATA_ERASURE,
            action="erase",
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            description=f"User data erased from: {', '.join(tables_deleted)}",
            metadata={"tables_deleted": tables_deleted},
            contains_pii=True,
            **kwargs,
        )


class AuditQueryService:
    """Query and filter audit logs.

    Provides advanced querying capabilities for audit logs.
    """

    def __init__(
        self,
        storage: AuditStorage,
        settings: Settings | None = None,
    ):
        self.storage = storage
        self.settings = settings or get_settings()

    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        severity: AuditSeverity | None = None,
        success: bool | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit logs with filters.

        Args:
            start_date: Start of date range
            end_date: End of date range
            event_types: Filter by event types
            user_id: Filter by user
            resource_type: Filter by resource type
            resource_id: Filter by specific resource
            severity: Filter by severity
            success: Filter by success/failure
            limit: Maximum results
            offset: Results offset

        Returns:
            List of matching audit entries
        """
        # Get base results
        entries = await self.storage.read_entries(
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            user_id=user_id,
            limit=limit + offset,  # Get extra for offset
        )

        # Apply additional filters
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]

        if resource_id:
            entries = [e for e in entries if e.resource_id == resource_id]

        if severity:
            entries = [e for e in entries if e.severity == severity]

        if success is not None:
            entries = [e for e in entries if e.success == success]

        # Apply offset and limit
        return entries[offset:offset + limit]

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get activity summary for a user."""
        start_date = datetime.utcnow() - timedelta(days=days)
        entries = await self.query(
            start_date=start_date,
            user_id=user_id,
            limit=10000,
        )

        # Aggregate activity
        activity = {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(entries),
            "by_event_type": {},
            "by_day": {},
            "failed_events": 0,
            "first_activity": None,
            "last_activity": None,
        }

        for entry in entries:
            # By event type
            event_type = entry.event_type.value
            activity["by_event_type"][event_type] = activity["by_event_type"].get(event_type, 0) + 1

            # By day
            day = entry.timestamp.date().isoformat()
            activity["by_day"][day] = activity["by_day"].get(day, 0) + 1

            # Failed events
            if not entry.success:
                activity["failed_events"] += 1

            # First/last activity
            if activity["first_activity"] is None or entry.timestamp < activity["first_activity"]:
                activity["first_activity"] = entry.timestamp.isoformat()
            if activity["last_activity"] is None or entry.timestamp > activity["last_activity"]:
                activity["last_activity"] = entry.timestamp.isoformat()

        return activity

    async def get_security_events(
        self,
        hours: int = 24,
    ) -> list[AuditEntry]:
        """Get security-related events."""
        security_event_types = [
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.RATE_LIMIT_EXCEEDED,
            AuditEventType.INVALID_INPUT,
        ]

        return await self.query(
            start_date=datetime.utcnow() - timedelta(hours=hours),
            event_types=security_event_types,
        )


class ComplianceReporter:
    """Generate compliance reports from audit logs.

    Provides reports for GDPR, SOC2, and other compliance frameworks.
    """

    def __init__(
        self,
        query_service: AuditQueryService,
        settings: Settings | None = None,
    ):
        self.query = query_service
        self.settings = settings or get_settings()

    async def generate_gdpr_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Generate GDPR compliance report."""
        report = {
            "report_type": "GDPR Compliance",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {},
        }

        # Data subject requests
        dsr_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.DATA_SUBJECT_REQUEST],
        )
        report["sections"]["data_subject_requests"] = {
            "total": len(dsr_events),
            "by_type": {},
        }
        for event in dsr_events:
            req_type = event.action
            report["sections"]["data_subject_requests"]["by_type"][req_type] = (
                report["sections"]["data_subject_requests"]["by_type"].get(req_type, 0) + 1
            )

        # Data erasure
        erasure_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.DATA_ERASURE],
        )
        report["sections"]["data_erasure"] = {
            "total": len(erasure_events),
            "users_affected": len(set(e.user_id for e in erasure_events if e.user_id)),
        }

        # Consent changes
        consent_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[
                AuditEventType.CONSENT_GRANTED,
                AuditEventType.CONSENT_WITHDRAWN,
            ],
        )
        report["sections"]["consent"] = {
            "total": len(consent_events),
            "granted": len([e for e in consent_events if e.event_type == AuditEventType.CONSENT_GRANTED]),
            "withdrawn": len([e for e in consent_events if e.event_type == AuditEventType.CONSENT_WITHDRAWN]),
        }

        # Data access
        access_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.DOCUMENT_ACCESSED],
            limit=10000,
        )
        pii_access = [e for e in access_events if e.contains_pii]
        report["sections"]["data_access"] = {
            "total_access_events": len(access_events),
            "pii_access_events": len(pii_access),
        }

        return report

    async def generate_access_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Generate access control report."""
        report = {
            "report_type": "Access Control",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {},
        }

        # Login events
        login_success = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.LOGIN_SUCCESS],
        )
        login_failure = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.LOGIN_FAILURE],
        )
        report["sections"]["authentication"] = {
            "successful_logins": len(login_success),
            "failed_logins": len(login_failure),
            "failure_rate": len(login_failure) / (len(login_success) + len(login_failure))
            if (len(login_success) + len(login_failure)) > 0 else 0,
        }

        # Access denied events
        access_denied = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.ACCESS_DENIED],
        )
        report["sections"]["authorization"] = {
            "access_denied_count": len(access_denied),
            "by_resource_type": {},
        }
        for event in access_denied:
            rt = event.resource_type or "unknown"
            report["sections"]["authorization"]["by_resource_type"][rt] = (
                report["sections"]["authorization"]["by_resource_type"].get(rt, 0) + 1
            )

        # Role changes
        role_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=[
                AuditEventType.ROLE_ASSIGNED,
                AuditEventType.ROLE_REVOKED,
            ],
        )
        report["sections"]["role_changes"] = {
            "total": len(role_events),
            "assigned": len([e for e in role_events if e.event_type == AuditEventType.ROLE_ASSIGNED]),
            "revoked": len([e for e in role_events if e.event_type == AuditEventType.ROLE_REVOKED]),
        }

        return report

    async def generate_security_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Generate security incident report."""
        report = {
            "report_type": "Security Incidents",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {},
        }

        # Security events
        security_types = [
            AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.RATE_LIMIT_EXCEEDED,
            AuditEventType.INVALID_INPUT,
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.ACCESS_DENIED,
        ]
        security_events = await self.query.query(
            start_date=start_date,
            end_date=end_date,
            event_types=security_types,
            limit=10000,
        )

        report["sections"]["summary"] = {
            "total_security_events": len(security_events),
            "by_type": {},
            "by_severity": {},
        }

        for event in security_events:
            # By type
            et = event.event_type.value
            report["sections"]["summary"]["by_type"][et] = (
                report["sections"]["summary"]["by_type"].get(et, 0) + 1
            )

            # By severity
            sev = event.severity.value
            report["sections"]["summary"]["by_severity"][sev] = (
                report["sections"]["summary"]["by_severity"].get(sev, 0) + 1
            )

        # Top offenders (users with most security events)
        user_counts: dict[str, int] = {}
        for event in security_events:
            if event.user_id:
                user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1

        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        report["sections"]["top_users"] = [
            {"user_id": uid, "event_count": count}
            for uid, count in top_users
        ]

        return report


# Module-level singletons
_audit_logger: AuditLogger | None = None
_audit_storage: AuditStorage | None = None


async def get_audit_logger(
    surrealdb_client: Any = None,
    log_dir: str | Path | None = None,
) -> AuditLogger:
    """Get or create audit logger."""
    global _audit_logger, _audit_storage
    if _audit_logger is None:
        _audit_storage = AuditStorage(
            log_dir=log_dir,
            surrealdb_client=surrealdb_client,
        )
        _audit_logger = AuditLogger(_audit_storage)
    return _audit_logger


async def get_audit_query_service() -> AuditQueryService:
    """Get audit query service."""
    global _audit_storage
    if _audit_storage is None:
        _audit_storage = AuditStorage()
    return AuditQueryService(_audit_storage)
