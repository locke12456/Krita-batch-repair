from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .repair_compat import qt_compat


ANNOTATION_KEY = "repair_plugin/state.json"
SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RepairStateRecord:
    canonical_layer_id: str
    export_key: str = ""
    group_id: str | None = None
    group_name: str | None = None
    active_layer_id: str = ""
    active_layer_name: str = ""
    deleted_layer_ids: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    refine_history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RepairStateRecord":
        return RepairStateRecord(
            canonical_layer_id=str(data.get("canonical_layer_id", "")),
            export_key=str(data.get("export_key", "")),
            group_id=data.get("group_id"),
            group_name=data.get("group_name"),
            active_layer_id=str(data.get("active_layer_id", "")),
            active_layer_name=str(data.get("active_layer_name", "")),
            deleted_layer_ids=[
                str(layer_id)
                for layer_id in list(data.get("deleted_layer_ids", []))
                if str(layer_id)
            ],
            replacements={
                str(old_id): str(new_id)
                for old_id, new_id in dict(data.get("replacements", {})).items()
                if str(old_id) and str(new_id)
            },
            refine_history=[
                dict(item)
                for item in list(data.get("refine_history", []))
                if isinstance(item, dict)
            ],
            updated_at=str(data.get("updated_at", "")),
        )


@dataclass
class RepairStateData:
    version: int = SCHEMA_VERSION
    records_by_canonical_layer_id: dict[str, RepairStateRecord] = field(default_factory=dict)
    records_by_export_key: dict[str, str] = field(default_factory=dict)
    records_by_group_id: dict[str, list[str]] = field(default_factory=dict)


class RepairStateStore:
    def __init__(self, document: Any, annotation_key: str = ANNOTATION_KEY) -> None:
        self.document = document
        self.annotation_key = annotation_key
        self.data = RepairStateData()
        self.load()

    def load(self) -> None:
        annotation = self.document.find_annotation(self.annotation_key)
        if annotation is None:
            self.data = RepairStateData()
            return

        payload = bytes(annotation).decode("utf-8")
        raw = self._migrate_raw(json.loads(payload))
        self.data = RepairStateData(version=int(raw.get("version", SCHEMA_VERSION)))

        for canonical_layer_id, record_data in raw.get("records_by_canonical_layer_id", {}).items():
            record = RepairStateRecord.from_dict(record_data)
            if not record.canonical_layer_id:
                record.canonical_layer_id = str(canonical_layer_id)
            if record.canonical_layer_id:
                self.data.records_by_canonical_layer_id[record.canonical_layer_id] = record

        for export_key, canonical_layer_id in raw.get("records_by_export_key", {}).items():
            export_key = str(export_key)
            canonical_layer_id = str(canonical_layer_id)
            if export_key and canonical_layer_id:
                self.data.records_by_export_key[export_key] = canonical_layer_id

        for group_id, canonical_layer_ids in raw.get("records_by_group_id", {}).items():
            group_id = str(group_id)
            ids = [str(item) for item in list(canonical_layer_ids or []) if str(item)]
            if group_id and ids:
                self.data.records_by_group_id[group_id] = ids

        self._rebuild_indexes()

    def _migrate_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        version = int(raw.get("version", 0) or 0)
        if version <= 0:
            raw = dict(raw)
            raw["version"] = SCHEMA_VERSION

        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported repair state schema version {version}; expected {SCHEMA_VERSION}."
            )

        raw.setdefault("records_by_canonical_layer_id", {})
        raw.setdefault("records_by_export_key", {})
        raw.setdefault("records_by_group_id", {})
        return raw

    def save(self) -> None:
        self._rebuild_indexes()
        raw = {
            "version": self.data.version,
            "records_by_canonical_layer_id": {
                key: record.to_dict()
                for key, record in self.data.records_by_canonical_layer_id.items()
            },
            "records_by_export_key": dict(self.data.records_by_export_key),
            "records_by_group_id": {
                key: list(value)
                for key, value in self.data.records_by_group_id.items()
            },
        }
        payload = json.dumps(raw, indent=2, ensure_ascii=False).encode("utf-8")
        self.document.annotate(self.annotation_key, qt_compat.QByteArray(payload))

    def upsert_record(self, record: RepairStateRecord) -> RepairStateRecord:
        if not record.canonical_layer_id:
            raise ValueError("canonical_layer_id is required")
        if not record.updated_at:
            record.updated_at = utc_timestamp()
        self.data.records_by_canonical_layer_id[record.canonical_layer_id] = record
        self.save()
        return record

    def resolve_by_canonical_layer_id(self, canonical_layer_id: str | None) -> RepairStateRecord | None:
        if not canonical_layer_id:
            return None
        return self.data.records_by_canonical_layer_id.get(str(canonical_layer_id))

    def resolve_by_export_key(self, export_key: str | None) -> RepairStateRecord | None:
        if not export_key:
            return None
        canonical_layer_id = self.data.records_by_export_key.get(str(export_key))
        return self.resolve_by_canonical_layer_id(canonical_layer_id)

    def resolve_group_records(self, group_id: str | None) -> list[RepairStateRecord]:
        if not group_id:
            return []
        result: list[RepairStateRecord] = []
        for canonical_layer_id in self.data.records_by_group_id.get(str(group_id), []):
            record = self.resolve_by_canonical_layer_id(canonical_layer_id)
            if record is not None:
                result.append(record)
        return result

    def resolve_replacement(self, old_layer_id: str | None) -> str | None:
        if not old_layer_id:
            return None

        target_id = str(old_layer_id)
        for record in self.data.records_by_canonical_layer_id.values():
            current = record.replacements.get(target_id)
            seen = {target_id}
            while current and current not in seen:
                seen.add(current)
                next_id = record.replacements.get(current)
                if not next_id:
                    return current
                current = next_id
        return None

    def record_refine_success(
        self,
        canonical_layer_id: str,
        old_layer_id: str,
        new_layer_id: str,
        *,
        export_key: str = "",
        group_id: str | None = None,
        group_name: str | None = None,
        active_layer_name: str = "",
        job_id: str = "",
        seed: int | None = None,
    ) -> RepairStateRecord:
        if not canonical_layer_id:
            raise ValueError("canonical_layer_id is required")
        if not old_layer_id:
            raise ValueError("old_layer_id is required")
        if not new_layer_id:
            raise ValueError("new_layer_id is required")

        record = self.resolve_by_canonical_layer_id(canonical_layer_id)
        if record is None:
            record = RepairStateRecord(canonical_layer_id=canonical_layer_id)

        if export_key:
            record.export_key = export_key
        if group_id:
            record.group_id = group_id
        if group_name:
            record.group_name = group_name

        if old_layer_id not in record.deleted_layer_ids:
            record.deleted_layer_ids.append(old_layer_id)
        record.replacements[old_layer_id] = new_layer_id
        record.active_layer_id = new_layer_id
        record.active_layer_name = active_layer_name
        record.updated_at = utc_timestamp()
        record.refine_history.append(
            {
                "old_layer_id": old_layer_id,
                "new_layer_id": new_layer_id,
                "job_id": job_id,
                "seed": seed,
                "updated_at": record.updated_at,
            }
        )
        return self.upsert_record(record)

    def all_records(self) -> list[RepairStateRecord]:
        return list(self.data.records_by_canonical_layer_id.values())

    def _rebuild_indexes(self) -> None:
        records_by_export_key: dict[str, str] = {}
        records_by_group_id: dict[str, list[str]] = {}

        for canonical_layer_id, record in self.data.records_by_canonical_layer_id.items():
            if record.export_key:
                records_by_export_key[record.export_key] = canonical_layer_id
            if record.group_id:
                records_by_group_id.setdefault(record.group_id, [])
                if canonical_layer_id not in records_by_group_id[record.group_id]:
                    records_by_group_id[record.group_id].append(canonical_layer_id)

        self.data.records_by_export_key = records_by_export_key
        self.data.records_by_group_id = records_by_group_id