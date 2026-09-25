from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClinicalRecord:
    """Normalized record used by training, retrieval, and sample data."""

    record_id: str
    split: str
    label: int
    query_text: str
    retrieval_text: str
    patient_id: str = ""
    center: str = ""
    secondary_endpoint: int | None = None
    diagnosis_teacher_row_id: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must not be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 or 1")
        if not self.query_text.strip() or not self.retrieval_text.strip():
            raise ValueError("query_text and retrieval_text must not be empty")
        if self.secondary_endpoint not in {None, 0, 1}:
            raise ValueError("secondary_endpoint must be 0, 1, or null")
