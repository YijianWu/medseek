from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from medseek.data.records import ClinicalRecord


def validate_dataset(records: Sequence[ClinicalRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("dataset must not be empty")
    record_ids = [record.record_id for record in records]
    duplicate_ids = [key for key, count in Counter(record_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate record_id values: {duplicate_ids[:5]}")

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.patient_id:
            patient_splits[record.patient_id].add(record.split)
    leaked = sorted(patient for patient, splits in patient_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"patients occur across splits: {leaked[:5]}")

    split_counts = Counter(record.split for record in records)
    label_counts = Counter(record.label for record in records)
    center_counts = Counter(record.center or "unknown" for record in records)
    return {
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "labels": {str(key): value for key, value in sorted(label_counts.items())},
        "centers": dict(sorted(center_counts.items())),
        "secondary_endpoint_records": sum(record.secondary_endpoint is not None for record in records),
        "diagnosis_supervision_records": sum(record.diagnosis_teacher_row_id is not None for record in records),
    }
