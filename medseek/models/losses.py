from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss used by the MedSeek risk encoder."""

    def __init__(self, temperature: float = 0.2, scale_by_temperature: bool = True) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.scale_by_temperature = bool(scale_by_temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, views, dimensions]")
        batch_size, views, dimensions = features.shape
        labels = labels.reshape(-1)
        if labels.shape[0] != batch_size:
            raise ValueError("labels and features have different batch sizes")

        contrast = F.normalize(features, dim=-1).reshape(batch_size * views, dimensions)
        temperature = max(self.temperature, 1.0e-6)
        logits = contrast @ contrast.T / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        class_mask = (labels[:, None] == labels[None, :]).to(logits.dtype)
        positive_mask = class_mask.repeat_interleave(views, 0).repeat_interleave(views, 1)
        self_mask = torch.eye(batch_size * views, device=logits.device, dtype=logits.dtype)
        positive_mask = positive_mask * (1.0 - self_mask)
        valid_mask = 1.0 - self_mask
        log_probability = logits - torch.log((torch.exp(logits) * valid_mask).sum(1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(1)
        valid = positive_count > 0
        if not torch.any(valid):
            return features.sum() * 0.0
        loss = -((positive_mask[valid] * log_probability[valid]).sum(1) / positive_count[valid])
        if self.scale_by_temperature:
            loss = loss * temperature
        return loss.mean()


def multi_positive_infonce(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE where records sharing a risk label are positives."""
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    doc_embeddings = F.normalize(doc_embeddings, dim=-1)
    logits = query_embeddings @ doc_embeddings.T / max(float(temperature), 1.0e-6)
    positive = labels[:, None] == labels[None, :]

    def direction_loss(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(scores, dim=1)
        weights = mask.to(log_prob.dtype)
        return -((weights * log_prob).sum(1) / weights.sum(1).clamp_min(1.0)).mean()

    return 0.5 * (direction_loss(logits, positive) + direction_loss(logits.T, positive.T))


def balanced_margin_logits(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    bucket_labels: torch.Tensor,
    *,
    top_k: int,
    exclude_diagonal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced positive-minus-negative Top-K similarity margins and valid row mask."""
    scores = F.normalize(query_embeddings, dim=-1) @ F.normalize(doc_embeddings, dim=-1).T
    rows, columns = scores.shape
    valid_docs = torch.ones_like(scores, dtype=torch.bool)
    if exclude_diagonal and rows == columns:
        valid_docs &= ~torch.eye(rows, device=scores.device, dtype=torch.bool)
    labels = bucket_labels.to(scores.device).reshape(-1)
    positive_k = max(1, int(top_k) // 2)
    negative_k = max(1, int(top_k) - positive_k)
    margins: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    for row in range(rows):
        positive = scores[row][valid_docs[row] & (labels == 1)]
        negative = scores[row][valid_docs[row] & (labels == 0)]
        valid = positive.numel() > 0 and negative.numel() > 0
        valid_rows.append(valid)
        if valid:
            pos = torch.topk(positive, min(positive_k, positive.numel())).values.mean()
            neg = torch.topk(negative, min(negative_k, negative.numel())).values.mean()
            margins.append(pos - neg)
        else:
            margins.append(scores[row].sum() * 0.0)
    return torch.stack(margins), torch.tensor(valid_rows, device=scores.device, dtype=torch.bool)


def diagnosis_prototype_loss(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    temperature: float = 0.2,
    teacher_temperature: float = 0.1,
    semantic_mix_alpha: float = 0.2,
) -> torch.Tensor:
    prototypes = F.normalize(prototype_embeddings, dim=-1)
    row_ids = row_ids.long()
    valid = (row_ids >= 0) & (row_ids < prototypes.shape[0])
    if not torch.any(valid):
        return query_embeddings.sum() * 0.0
    row_ids = row_ids[valid]
    query = F.normalize(query_embeddings[valid], dim=-1)
    docs = F.normalize(doc_embeddings[valid], dim=-1)
    query_logits = query @ prototypes.T / max(float(temperature), 1e-6)
    doc_logits = docs @ prototypes.T / max(float(temperature), 1e-6)
    one_hot = F.one_hot(row_ids, num_classes=prototypes.shape[0]).to(query_logits.dtype)
    semantic = torch.softmax(
        prototypes[row_ids] @ prototypes.T / max(float(teacher_temperature), 1e-6), dim=1
    )
    targets = (1.0 - semantic_mix_alpha) * one_hot + semantic_mix_alpha * semantic
    query_loss = -(targets * F.log_softmax(query_logits, dim=1)).sum(1).mean()
    doc_loss = -(targets * F.log_softmax(doc_logits, dim=1)).sum(1).mean()
    return 0.5 * (query_loss + doc_loss)
