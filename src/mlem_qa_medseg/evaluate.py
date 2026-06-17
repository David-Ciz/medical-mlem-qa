from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_params, resolve_path
from .data import SegmentationDataset
from .device import get_device, supports_pin_memory
from .metrics import dice_score, iou_score, precision_score, recall_score
from .models import MODEL_NAMES, build_model


def load_checkpoint(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(params_path: Path, model_name: str, split: str) -> None:
    params = load_params(params_path)
    data_params = params["data"]
    train_params = params["train"]
    processed_dir = resolve_path(data_params["processed_dir"])
    checkpoint_path = resolve_path(params["models"][model_name]["output_dir"]) / "best.pt"
    report_dir = resolve_path("reports") / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    model = load_checkpoint(model_name, checkpoint_path, device)
    dataset = SegmentationDataset(
        processed_dir / "splits" / f"{split}.csv",
        image_size=int(data_params["image_size"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_params["batch_size"]),
        shuffle=False,
        num_workers=int(train_params["num_workers"]),
        pin_memory=supports_pin_memory(device),
    )
    threshold = float(train_params["threshold"])
    rows: list[dict[str, float | str]] = []

    for batch in tqdm(loader, desc=f"evaluate {model_name}/{split}"):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(image))
        pred = (prob >= threshold).float()
        batch_dice = dice_score(pred, mask).cpu()
        batch_iou = iou_score(pred, mask).cpu()
        batch_precision = precision_score(pred, mask).cpu()
        batch_recall = recall_score(pred, mask).cpu()
        for idx, image_path in enumerate(batch["image_path"]):
            rows.append(
                {
                    "image_path": image_path,
                    "mask_path": batch["mask_path"][idx],
                    "dice": float(batch_dice[idx]),
                    "iou": float(batch_iou[idx]),
                    "precision": float(batch_precision[idx]),
                    "recall": float(batch_recall[idx]),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics_path = report_dir / f"{split}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "model": model_name,
        "split": split,
        "n": int(len(metrics)),
        "dice_mean": float(metrics["dice"].mean()),
        "iou_mean": float(metrics["iou"].mean()),
        "precision_mean": float(metrics["precision"].mean()),
        "recall_mean": float(metrics["recall"].mean()),
    }
    summary_path = report_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.params, args.model, args.split)


if __name__ == "__main__":
    main()
