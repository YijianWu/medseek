from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from medseek.data.records import ClinicalRecord


REQUIRED_COLUMNS = {"record_id", "split", "query_text", "retrieval_text"}
_LABEL_COLUMNS = {"label", "label_high_risk"}


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _label_from_mapping(row: Mapping[str, object]) -> int:
    values: dict[str, int] = {}
    for name in sorted(_LABEL_COLUMNS):
        value = row.get(name)
        if not _is_missing(value):
            values[name] = int(value)

    if not values:
        return -1
    if len(set(values.values())) > 1:
        raise ValueError(
            "conflicting label values: "
            + ", ".join(f"{name}={value}" for name, value in values.items())
        )
    return next(iter(values.values()))


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = int(value)
    if parsed not in {0, 1}:
        raise ValueError(f"optional binary value must be 0 or 1, got {value!r}")
    return parsed


def record_from_mapping(row: Mapping[str, object], *, row_number: int | None = None) -> ClinicalRecord:
    where = "" if row_number is None else f" at row {row_number}"
    try:
        record_id = str(row.get("record_id") or row.get("ehr_id") or "").strip()
        split = str(row.get("split") or "").strip().lower()
        label = _label_from_mapping(row)
        query_text = str(row.get("query_text") or "").strip()
        retrieval_text = str(row.get("retrieval_text") or "").strip()
        teacher_id = row.get("diagnosis_teacher_row_id")
        metadata = row.get("metadata", {})
        if isinstance(metadata, str) and metadata.strip():
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            metadata = {}
        return ClinicalRecord(
            record_id=record_id,
            split=split,
            label=label,
            query_text=query_text,
            retrieval_text=retrieval_text,
            patient_id=str(row.get("patient_id") or "").strip(),
            center=str(row.get("center") or row.get("site") or "").strip(),
            secondary_endpoint=_optional_int(row.get("secondary_endpoint")),
            diagnosis_teacher_row_id=None if teacher_id in {None, ""} else int(teacher_id),
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid clinical record{where}: {exc}") from exc


def load_csv_records(path: str | Path) -> list[ClinicalRecord]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if not _LABEL_COLUMNS.intersection(fieldnames):
            raise ValueError(f"CSV must contain at least one of: {sorted(_LABEL_COLUMNS)}")
        records = [record_from_mapping(row, row_number=index) for index, row in enumerate(reader, start=2)]
    if not records:
        raise ValueError(f"no records found in {source}")
    return records


def load_jsonl_records(path: str | Path) -> list[ClinicalRecord]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[ClinicalRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {row_number} must be an object")
            records.append(record_from_mapping(payload, row_number=row_number))
    if not records:
        raise ValueError(f"no records found in {source}")
    return records


def load_records(path: str | Path) -> list[ClinicalRecord]:
    source = Path(path)
    return load_jsonl_records(source) if source.suffix.lower() == ".jsonl" else load_csv_records(source)


def write_jsonl_records(records: Iterable[ClinicalRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "record_id": record.record_id,
                "patient_id": record.patient_id,
                "center": record.center,
                "split": record.split,
                "label_high_risk": record.label,
                "query_text": record.query_text,
                "retrieval_text": record.retrieval_text,
                "secondary_endpoint": record.secondary_endpoint,
                "diagnosis_teacher_row_id": record.diagnosis_teacher_row_id,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def split_records(records: list[ClinicalRecord]) -> tuple[list[ClinicalRecord], list[ClinicalRecord]]:
    train = [record for record in records if record.split == "train"]
    test = [record for record in records if record.split == "test"]
    if not train or not test:
        raise ValueError("data must contain both train and test records")
    if {record.label for record in train} != {0, 1}:
        raise ValueError("training records must contain both risk labels")
    if {record.label for record in test} != {0, 1}:
        raise ValueError("test records must contain both risk labels")
    return train, test
