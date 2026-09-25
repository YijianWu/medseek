from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from medseek.data import ContrastiveDataset, FixedLabelRatioBatchSampler, ClinicalRecord, collate_records
from medseek.models import MedSeekDualEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_loader(records: list[ClinicalRecord], config: dict[str, Any]) -> DataLoader:
    sampler = FixedLabelRatioBatchSampler(
        records,
        batch_size=int(config.get("batch_size", 64)),
        positive_ratio=float(config.get("risk_positive_ratio", 0.1)),
        positive_min_count=int(config.get("risk_positive_min_count", 2)),
        seed=int(config.get("seed", 42)),
        drop_last=False,
    )
    return DataLoader(
        ContrastiveDataset(records),
        batch_sampler=sampler,
        collate_fn=collate_records,
        num_workers=int(config.get("num_workers", 0)),
    )


def build_eval_loader(records: list[ClinicalRecord], config: dict[str, Any]) -> DataLoader:
    return DataLoader(
        ContrastiveDataset(records),
        batch_size=int(config.get("eval_batch_size", config.get("batch_size", 64))),
        shuffle=False,
        collate_fn=collate_records,
        num_workers=int(config.get("num_workers", 0)),
    )


@torch.inference_mode()
def evaluate_loss(model: MedSeekDualEncoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    for batch in loader:
        values.append(float(model(batch)["loss"].detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def train_model(
    model: MedSeekDualEncoder,
    train_records: list[ClinicalRecord],
    validation_records: list[ClinicalRecord],
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device)

    train_loader = build_train_loader(train_records, config)
    validation_loader = build_eval_loader(validation_records, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config.get("max_epochs", 3))
    gradient_clip = float(config.get("gradient_clip_norm", 1.0))
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.current_epoch = epoch
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        model.train()
        train_values: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        validation_loss = evaluate_loss(model, validation_loader, device)
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_values)),
            "validation_loss": validation_loss,
        }
        history.append(epoch_summary)
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save_pretrained(output / "best-checkpoint")
        print(json.dumps(epoch_summary))

    summary = {
        "epochs": epochs,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "best_validation_loss": best_loss,
        "checkpoint": str(output / "best-checkpoint"),
        "history": history,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
