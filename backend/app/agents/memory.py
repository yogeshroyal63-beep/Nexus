"""
Memory Agent — persistent, cross-session incident history.

Two backends:
  - LocalJSONMemoryStore: zero-setup, file-backed. Default for dev/demo.
  - DynamoDBMemoryStore: AWS-native, scales to production. Set MEMORY_BACKEND=dynamodb.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import settings
from app.models.schemas import IncidentRecord

logger = logging.getLogger(__name__)


class BaseMemoryStore(ABC):
    @abstractmethod
    async def save_incident(self, record: IncidentRecord) -> None: ...

    @abstractmethod
    async def get_incident(self, incident_id: str) -> IncidentRecord | None: ...

    @abstractmethod
    async def find_similar(self, model_urn: str, limit: int = 5) -> list[IncidentRecord]: ...

    @abstractmethod
    async def all_incidents(self, limit: int = 100) -> list[IncidentRecord]: ...

    @abstractmethod
    async def claim_incident_for_approval(self, incident_id: str) -> IncidentRecord | None: ...


class LocalJSONMemoryStore(BaseMemoryStore):
    """File-backed store. Zero setup, works fully offline."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.MEMORY_LOCAL_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")
        self._lock = asyncio.Lock()

    def _read_all(self) -> list[dict]:
        try:
            content = self.path.read_text()
            if not content.strip():
                return []
            return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            logger.error("Memory file corrupt or missing (%s); treating as empty.", exc)
            return []

    def _write_all(self, records: list[dict]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
        tmp_path.write_text(json.dumps(records, indent=2, default=str))
        os.replace(tmp_path, self.path)

    async def save_incident(self, record: IncidentRecord) -> None:
        async with self._lock:
            records = self._read_all()
            payload = json.loads(record.model_dump_json())
            records = [r for r in records if r.get("incident_id") != record.incident_id]
            records.append(payload)
            self._write_all(records)

    async def get_incident(self, incident_id: str) -> IncidentRecord | None:
        async with self._lock:
            records = self._read_all()
        for r in records:
            if r.get("incident_id") == incident_id:
                return IncidentRecord(**r)
        return None

    async def find_similar(self, model_urn: str, limit: int = 5) -> list[IncidentRecord]:
        async with self._lock:
            all_records = self._read_all()
        records = [r for r in all_records if r.get("model_urn") == model_urn]
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [IncidentRecord(**r) for r in records[:limit]]

    async def all_incidents(self, limit: int = 100) -> list[IncidentRecord]:
        async with self._lock:
            records = self._read_all()
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [IncidentRecord(**r) for r in records[:limit]]

    async def claim_incident_for_approval(self, incident_id: str) -> IncidentRecord | None:
        async with self._lock:
            records = self._read_all()
            for r in records:
                if r.get("incident_id") != incident_id:
                    continue
                if r.get("plan") is None:
                    return None
                if r.get("outcome") is not None:
                    return None
                if r.get("_approval_claimed"):
                    return None
                r["_approval_claimed"] = True
                self._write_all(records)
                claimed = dict(r)
                claimed.pop("_approval_claimed", None)
                return IncidentRecord(**claimed)
            return None


class DynamoDBMemoryStore(BaseMemoryStore):
    """
    AWS DynamoDB-backed store. Production-grade, multi-instance safe.
    Requires MEMORY_BACKEND=dynamodb and valid AWS credentials.
    """

    def __init__(self):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 not installed. pip install boto3") from exc

        self._client = boto3.client(
            "dynamodb",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
        )
        self._table = settings.DYNAMODB_TABLE
        self._lock = asyncio.Lock()

    def _serialize(self, record: IncidentRecord) -> dict:
        data = json.loads(record.model_dump_json())
        return {"incident_id": {"S": record.incident_id}, "data": {"S": json.dumps(data, default=str)}}

    def _deserialize(self, item: dict) -> IncidentRecord:
        data = json.loads(item["data"]["S"])
        return IncidentRecord(**data)

    async def save_incident(self, record: IncidentRecord) -> None:
        loop = asyncio.get_running_loop()
        item = self._serialize(record)
        await loop.run_in_executor(
            None,
            lambda: self._client.put_item(TableName=self._table, Item=item),
        )

    async def get_incident(self, incident_id: str) -> IncidentRecord | None:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.get_item(
                TableName=self._table,
                Key={"incident_id": {"S": incident_id}},
            ),
        )
        item = response.get("Item")
        if not item:
            return None
        return self._deserialize(item)

    async def find_similar(self, model_urn: str, limit: int = 5) -> list[IncidentRecord]:
        # Scan with filter — for production add a GSI on model_urn
        all_records = await self.all_incidents(limit=500)
        filtered = [r for r in all_records if r.model_urn == model_urn]
        return filtered[:limit]

    async def all_incidents(self, limit: int = 100) -> list[IncidentRecord]:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.scan(TableName=self._table, Limit=limit),
        )
        items = response.get("Items", [])
        records = []
        for item in items:
            try:
                records.append(self._deserialize(item))
            except Exception as exc:
                logger.warning("Could not deserialize DynamoDB item: %s", exc)
        records.sort(key=lambda r: str(r.created_at), reverse=True)
        return records

    async def claim_incident_for_approval(self, incident_id: str) -> IncidentRecord | None:
        # DynamoDB conditional update for atomic claim
        loop = asyncio.get_running_loop()
        async with self._lock:
            record = await self.get_incident(incident_id)
            if record is None or record.plan is None or record.outcome is not None:
                return None
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._client.update_item(
                        TableName=self._table,
                        Key={"incident_id": {"S": incident_id}},
                        UpdateExpression="SET claimed = :c",
                        ConditionExpression="attribute_not_exists(claimed)",
                        ExpressionAttributeValues={":c": {"BOOL": True}},
                    ),
                )
                return record
            except self._client.exceptions.ConditionalCheckFailedException:
                return None


_store_singleton: BaseMemoryStore | None = None


def get_memory_store() -> BaseMemoryStore:
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    backend = os.getenv("MEMORY_BACKEND", settings.MEMORY_BACKEND)
    if backend == "dynamodb":
        logger.info("Memory agent using DynamoDB backend (table=%s)", settings.DYNAMODB_TABLE)
        _store_singleton = DynamoDBMemoryStore()
    else:
        logger.info("Memory agent using local JSON backend at %s", settings.MEMORY_LOCAL_PATH)
        _store_singleton = LocalJSONMemoryStore()
    return _store_singleton
