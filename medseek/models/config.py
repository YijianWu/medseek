from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class MedSeekConfig:
    model_name_or_path: str = "intfloat/multilingual-e5-large"
    max_length: int = 512
    temperature: float = 0.2
    risk_loss_mode: str = "supcon"
    risk_loss_weight: float = 1.0
    diagnosis_aux_enabled: bool = True
    diagnosis_loss_weight: float = 0.1
    diagnosis_temperature: float = 0.2
    diagnosis_teacher_temperature: float = 0.1
    diagnosis_semantic_mix_alpha: float = 0.2
    diagnosis_warmup_epochs: int = 1
    diagnosis_prototype_path: str | None = None
    secondary_endpoint_cls_enabled: bool = True
    secondary_endpoint_cls_loss_weight: float = 0.2
    secondary_endpoint_dropout: float = 0.1
    secondary_endpoint_pos_weight: float | None = None
    secondary_endpoint_margin_enabled: bool = True
    secondary_endpoint_margin_weight: float = 0.01
    secondary_endpoint_margin_top_k: int = 20
    inference_margin_weight: float = 0.0
    inference_margin_top_k: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MedSeekConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in known})
