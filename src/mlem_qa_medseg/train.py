from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_params, resolve_path
from .data import SegmentationDataset
from .device import get_device, supports_pin_memory
from .metrics import dice_from_logits, dice_loss
from .models import MODEL_NAMES, build_model


def make_loader(
    split_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = SegmentationDataset(split_csv, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    dice_scores: list[torch.Tensor] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(image)
        loss = combined_loss(logits, mask)
        losses.append(float(loss.item()))
        dice_scores.append(dice_from_logits(logits, mask, threshold=threshold).cpu())

    dice = torch.cat(dice_scores).mean().item() if dice_scores else 0.0
    loss = sum(losses) / max(len(losses), 1)
    return {"loss": loss, "dice": dice}


def train(params_path: Path, model_name: str) -> None:
    params = load_params(params_path)
    data_params = params["data"]
    train_params = params["train"]
    output_dir = resolve_path(params["models"][model_name]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(params["seed"]))
    device = get_device()
    print(f"Using device: {device}")
    image_size = int(data_params["image_size"])
    processed_dir = resolve_path(data_params["processed_dir"])
    train_loader = make_loader(
        processed_dir / "splits" / "train.csv",
        image_size=image_size,
        batch_size=int(train_params["batch_size"]),
        num_workers=int(train_params["num_workers"]),
        augment=True,
        shuffle=True,
        pin_memory=supports_pin_memory(device),
    )
    val_loader = make_loader(
        processed_dir / "splits" / "val.csv",
        image_size=image_size,
        batch_size=int(train_params["batch_size"]),
        num_workers=int(train_params["num_workers"]),
        augment=False,
        shuffle=False,
        pin_memory=supports_pin_memory(device),
    )

    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_params["learning_rate"]),
        weight_decay=float(train_params["weight_decay"]),
    )
    use_amp = bool(train_params["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_dice = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_params["epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss = combined_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            progress.set_postfix(loss=sum(epoch_losses) / len(epoch_losses))

        val_metrics = validate(
            model,
            val_loader,
            device=device,
            threshold=float(train_params["threshold"]),
        )
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True))

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model": model_name,
                    "state_dict": model.state_dict(),
                    "params": params,
                    "best_val_dice": best_dice,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.params, args.model)


if __name__ == "__main__":
    main()
