from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseek.models.config import MedSeekConfig
from medseek.models.e5 import TrainableE5Encoder
from medseek.models.losses import (
    SupervisedContrastiveLoss,
    balanced_margin_logits,
    diagnosis_prototype_loss,
    multi_positive_infonce,
)


class MedSeekDualEncoder(nn.Module):
    """Compact implementation of the MedSeek multi-objective dual encoder."""

    def __init__(self, config: MedSeekConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder or TrainableE5Encoder(config.model_name_or_path, config.max_length)
        hidden_size = int(getattr(self.encoder, "hidden_size"))
        self.risk_loss = SupervisedContrastiveLoss(config.temperature)
        self.secondary_classifier = (
            nn.Sequential(nn.Dropout(config.secondary_endpoint_dropout), nn.Linear(hidden_size, 1))
            if config.secondary_endpoint_cls_enabled
            else None
        )
        self.register_buffer("diagnosis_prototypes", torch.empty(0, hidden_size), persistent=False)
        self.current_epoch = 0
        if config.diagnosis_prototype_path:
            self.load_diagnosis_prototypes(config.diagnosis_prototype_path)

    def load_diagnosis_prototypes(self, path: str | Path) -> None:
        values = np.load(path).astype("float32")
        if values.ndim != 2 or values.shape[1] != self.diagnosis_prototypes.shape[1]:
            raise ValueError("diagnosis prototype dimensions do not match the encoder")
        self.diagnosis_prototypes = torch.from_numpy(values)

    def _risk_component(
        self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        mode = self.config.risk_loss_mode.lower().replace("-", "_")
        supcon = self.risk_loss(torch.stack((query_embeddings, doc_embeddings), dim=1), labels)
        if mode in {"supcon", "in_batch_supervised_contrastive"}:
            return supcon
        infonce = multi_positive_infonce(query_embeddings, doc_embeddings, labels, self.config.temperature)
        if mode in {"batch_infonce", "infonce"}:
            return infonce
        if mode in {"supcon_infonce", "combined"}:
            return 0.5 * (supcon + infonce)
        raise ValueError(f"unsupported risk_loss_mode: {self.config.risk_loss_mode}")

    @staticmethod
    def _margin_loss(
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        bucket_labels: torch.Tensor,
        target_labels: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        margins, valid = balanced_margin_logits(
            query_embeddings, doc_embeddings, bucket_labels, top_k=top_k, exclude_diagonal=True
        )
        if not torch.any(valid):
            return query_embeddings.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            margins[valid] / max(float(temperature), 1e-6), target_labels[valid].float()
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        query_embeddings = self.encoder(batch["query_texts"], role="query")
        doc_embeddings = self.encoder(batch["doc_texts"], role="document")
        device = query_embeddings.device
        labels = batch["labels"].to(device).long()

        components: dict[str, torch.Tensor] = {
            "risk": self._risk_component(query_embeddings, doc_embeddings, labels)
        }
        zero = query_embeddings.sum() * 0.0

        if self.config.inference_margin_weight > 0:
            components["inference_margin"] = self._margin_loss(
                query_embeddings,
                doc_embeddings,
                labels,
                labels,
                top_k=self.config.inference_margin_top_k,
                temperature=self.config.temperature,
            )
        else:
            components["inference_margin"] = zero

        diagnosis_ids = batch.get("diagnosis_teacher_row_ids")
        diagnosis_ready = (
            self.config.diagnosis_aux_enabled
            and self.current_epoch >= self.config.diagnosis_warmup_epochs
            and self.diagnosis_prototypes.numel() > 0
            and diagnosis_ids is not None
        )
        components["diagnosis"] = (
            diagnosis_prototype_loss(
                query_embeddings,
                doc_embeddings,
                self.diagnosis_prototypes.to(device),
                diagnosis_ids.to(device),
                temperature=self.config.diagnosis_temperature,
                teacher_temperature=self.config.diagnosis_teacher_temperature,
                semantic_mix_alpha=self.config.diagnosis_semantic_mix_alpha,
            )
            if diagnosis_ready
            else zero
        )

        secondary_labels = batch.get("secondary_endpoint_labels")
        secondary_mask = batch.get("secondary_endpoint_mask")
        if secondary_labels is not None and secondary_mask is not None:
            secondary_labels = secondary_labels.to(device).float()
            secondary_mask = secondary_mask.to(device).bool()
        if self.secondary_classifier is not None and secondary_mask is not None and torch.any(secondary_mask):
            logits = self.secondary_classifier(query_embeddings[secondary_mask]).squeeze(-1)
            pos_weight = self.config.secondary_endpoint_pos_weight
            weight = None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
            components["secondary_cls"] = F.binary_cross_entropy_with_logits(
                logits, secondary_labels[secondary_mask], pos_weight=weight
            )
        else:
            components["secondary_cls"] = zero

        if (
            self.config.secondary_endpoint_margin_enabled
            and secondary_mask is not None
            and int(secondary_mask.sum()) >= 3
            and torch.unique(secondary_labels[secondary_mask]).numel() == 2
        ):
            valid_query = query_embeddings[secondary_mask]
            valid_doc = doc_embeddings[secondary_mask]
            valid_labels = secondary_labels[secondary_mask].long()
            components["secondary_margin"] = self._margin_loss(
                valid_query,
                valid_doc,
                valid_labels,
                valid_labels,
                top_k=self.config.secondary_endpoint_margin_top_k,
                temperature=self.config.temperature,
            )
        else:
            components["secondary_margin"] = zero

        total = (
            self.config.risk_loss_weight * components["risk"]
            + self.config.inference_margin_weight * components["inference_margin"]
            + self.config.diagnosis_loss_weight * components["diagnosis"]
            + self.config.secondary_endpoint_cls_loss_weight * components["secondary_cls"]
            + self.config.secondary_endpoint_margin_weight * components["secondary_margin"]
        )
        return {
            "loss": total,
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
            **{f"loss_{key}": value for key, value in components.items()},
        }

    def save_pretrained(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        encoder_dir = target / "encoder"
        if not hasattr(self.encoder, "save_pretrained"):
            raise ValueError("encoder does not support save_pretrained")
        self.encoder.save_pretrained(encoder_dir)
        heads = {} if self.secondary_classifier is None else self.secondary_classifier.state_dict()
        torch.save(heads, target / "medseek_heads.pt")
        (target / "medseek_config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> "MedSeekDualEncoder":
        source = Path(checkpoint_dir)
        config = MedSeekConfig.from_mapping(json.loads((source / "medseek_config.json").read_text(encoding="utf-8")))
        config.diagnosis_prototype_path = None
        config.model_name_or_path = str(source / "encoder")
        model = cls(config)
        heads_path = source / "medseek_heads.pt"
        if model.secondary_classifier is not None and heads_path.is_file():
            model.secondary_classifier.load_state_dict(torch.load(heads_path, map_location="cpu", weights_only=True))
        return model
