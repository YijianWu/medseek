from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from medseek.data.records import ClinicalRecord


class FixedLabelRatioBatchSampler(Sampler[list[int]]):
    """Build patient-unique batches with a controlled high-risk fraction."""

    def __init__(
        self,
        records: Sequence[ClinicalRecord],
        batch_size: int,
        positive_ratio: float = 0.1,
        positive_min_count: int = 2,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if not 0.0 < positive_ratio < 1.0:
            raise ValueError("positive_ratio must be between 0 and 1")
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.positive_count = min(batch_size - 1, max(1, positive_min_count, round(batch_size * positive_ratio)))
        self.negative_count = batch_size - self.positive_count
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.by_label = {
            label: [index for index, record in enumerate(self.records) if record.label == label] for label in (0, 1)
        }
        if not all(self.by_label.values()):
            raise ValueError("sampler requires both risk labels")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.records) / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {label: values.copy() for label, values in self.by_label.items()}
        for values in pools.values():
            rng.shuffle(values)
        cursors = {0: 0, 1: 0}
        for _ in range(len(self)):
            batch: list[int] = []
            used_patients: set[str] = set()
            for label, count in ((1, self.positive_count), (0, self.negative_count)):
                attempts = 0
                while sum(self.records[index].label == label for index in batch) < count:
                    pool = pools[label]
                    index = pool[cursors[label] % len(pool)]
                    cursors[label] += 1
                    attempts += 1
                    patient = self.records[index].patient_id or self.records[index].record_id
                    if patient in used_patients and attempts <= len(pool) * 2:
                        continue
                    batch.append(index)
                    used_patients.add(patient)
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
