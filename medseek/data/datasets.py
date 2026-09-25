from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from medseek.data.records import ClinicalRecord


class ContrastiveDataset(Dataset[ClinicalRecord]):
    def __init__(self, records: Sequence[ClinicalRecord]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("contrastive dataset must not be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ClinicalRecord:
        return self.records[index]


def collate_records(items: list[ClinicalRecord]) -> dict[str, object]:
    secondary_mask = torch.tensor([item.secondary_endpoint is not None for item in items], dtype=torch.bool)
    secondary_labels = torch.tensor(
        [0 if item.secondary_endpoint is None else item.secondary_endpoint for item in items], dtype=torch.float32
    )
    diagnosis_ids = torch.tensor(
        [-1 if item.diagnosis_teacher_row_id is None else item.diagnosis_teacher_row_id for item in items],
        dtype=torch.long,
    )
    return {
        "record_ids": [item.record_id for item in items],
        "patient_ids": [item.patient_id for item in items],
        "centers": [item.center for item in items],
        "labels": torch.tensor([item.label for item in items], dtype=torch.long),
        "query_texts": [item.query_text for item in items],
        "doc_texts": [item.retrieval_text for item in items],
        "secondary_endpoint_labels": secondary_labels,
        "secondary_endpoint_mask": secondary_mask,
        "diagnosis_teacher_row_ids": diagnosis_ids,
    }
