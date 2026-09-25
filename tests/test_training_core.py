import torch
import torch.nn as nn
import torch.nn.functional as functional

from medseek.data import ClinicalRecord, FixedLabelRatioBatchSampler, collate_records, validate_dataset
from medseek.models import MedSeekConfig, MedSeekDualEncoder


class TinyTextEncoder(nn.Module):
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, self.hidden_size)

    def forward(self, texts, *, role):
        offset = 1 if role == "query" else 2
        ids = torch.tensor([(sum(map(ord, text)) + offset) % 257 for text in texts])
        return functional.normalize(self.embedding(ids), dim=1)


def make_records():
    return [
        ClinicalRecord(
            record_id=f"R{index}",
            patient_id=f"P{index}",
            split="train",
            label=index % 2,
            query_text=f"query {index}",
            retrieval_text=f"document {index}",
            secondary_endpoint=(index // 2) % 2,
            diagnosis_teacher_row_id=index % 3,
        )
        for index in range(8)
    ]


def test_fixed_ratio_sampler_and_validation():
    records = make_records()
    summary = validate_dataset(records)
    batch = next(iter(FixedLabelRatioBatchSampler(records, batch_size=4, positive_ratio=0.5)))
    assert summary["records"] == 8
    assert sum(records[index].label for index in batch) == 2
    assert len({records[index].patient_id for index in batch}) == 4


def test_multitask_dual_encoder_backward():
    records = make_records()[:6]
    config = MedSeekConfig(
        diagnosis_aux_enabled=True,
        diagnosis_warmup_epochs=0,
        secondary_endpoint_cls_enabled=True,
        secondary_endpoint_margin_enabled=True,
        inference_margin_weight=0.1,
    )
    model = MedSeekDualEncoder(config, encoder=TinyTextEncoder())
    model.diagnosis_prototypes = functional.normalize(torch.randn(3, 8), dim=1)
    output = model(collate_records(records))
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert output["loss_risk"].item() > 0
    assert model.encoder.embedding.weight.grad is not None
