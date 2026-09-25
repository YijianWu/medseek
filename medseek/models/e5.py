from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TrainableE5Encoder(nn.Module):
    """Differentiable shared E5 encoder with role-specific E5 prefixes."""

    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, texts: Sequence[str], *, role: str) -> torch.Tensor:
        if role not in {"query", "document"}:
            raise ValueError(f"unsupported encoder role: {role}")
        prefix = "query" if role == "query" else "passage"
        prepared = [f"{prefix}: {str(text).strip()}" for text in texts]
        if not prepared or any(text == f"{prefix}: " for text in prepared):
            raise ValueError("texts must contain non-empty values")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            prepared, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.model(**tokens)
        mask = tokens["attention_mask"][..., None].bool()
        hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
        pooled = hidden.sum(1) / tokens["attention_mask"].sum(1)[..., None]
        return F.normalize(pooled, dim=1)

    def save_pretrained(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        return target
