from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


class E5TextEncoder:
    """Minimal E5 encoder used by the public FAISS demonstration."""

    def __init__(self, model_name_or_path: str, *, device: str = "auto", max_length: int = 256) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = device
        self.max_length = int(max_length)
        checkpoint = Path(model_name_or_path)
        resolved_path = checkpoint / "encoder" if (checkpoint / "medseek_config.json").is_file() else model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        self.model = AutoModel.from_pretrained(resolved_path).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode_numpy(self, texts: Sequence[str], *, role: str) -> np.ndarray:
        if role not in {"query", "document"}:
            raise ValueError(f"unsupported encoder role: {role}")
        prefix = "query" if role == "query" else "passage"
        prepared = [f"{prefix}: {str(text).strip()}" for text in texts]
        if not prepared or any(text == f"{prefix}: " for text in prepared):
            raise ValueError("texts must contain non-empty values")
        tokens = self.tokenizer(
            prepared,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        output = self.model(**tokens)
        mask = tokens["attention_mask"][..., None].bool()
        hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
        pooled = hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
        return functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype("float32")

