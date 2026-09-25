from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from medseek.models.config import MedSeekConfig


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML object")
    for section in ("model", "training", "data", "evaluation"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"missing config section: {section}")
    return payload


def model_config_from_experiment(payload: dict[str, Any]) -> MedSeekConfig:
    model = dict(payload["model"])
    diagnosis = dict(payload.get("diagnosis_aux", {}))
    secondary = dict(payload.get("secondary_endpoint_aux", {}))
    merged = {
        **model,
        "diagnosis_aux_enabled": diagnosis.get("enabled", False),
        "diagnosis_loss_weight": diagnosis.get("loss_weight", 0.0),
        "diagnosis_temperature": diagnosis.get("temperature", 0.2),
        "diagnosis_teacher_temperature": diagnosis.get("teacher_temperature", 0.1),
        "diagnosis_semantic_mix_alpha": diagnosis.get("semantic_mix_alpha", 0.2),
        "diagnosis_warmup_epochs": diagnosis.get("warmup_epochs", 1),
        "diagnosis_prototype_path": diagnosis.get("prototype_path"),
        "secondary_endpoint_cls_enabled": secondary.get("enabled", False),
        "secondary_endpoint_cls_loss_weight": secondary.get("cls_loss_weight", 0.0),
        "secondary_endpoint_dropout": secondary.get("dropout", 0.1),
        "secondary_endpoint_pos_weight": secondary.get("pos_weight"),
        "secondary_endpoint_margin_enabled": secondary.get("margin_enabled", False),
        "secondary_endpoint_margin_weight": secondary.get("margin_weight", 0.0),
        "secondary_endpoint_margin_top_k": secondary.get("margin_top_k", 20),
    }
    return MedSeekConfig.from_mapping(merged)
