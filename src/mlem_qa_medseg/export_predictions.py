from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_params, resolve_path
from .data import SegmentationDataset
from .device import get_device, supports_pin_memory
from .evaluate import load_checkpoint
from .metrics import dice_score, iou_score, precision_score, recall_score
from .models import MODEL_NAMES


@torch.no_grad()
def export_predictions(params_path: Path, model_name: str, split: str) -> None:
    params = load_params(params_path)
    data_params = params["data"]
    train_params = params["train"]
    processed_dir = resolve_path(data_params["processed_dir"])
    checkpoint_path = resolve_path(params["models"][model_name]["output_dir"]) / "best.pt"
    output_dir = resolve_path("predictions") / model_name / split
    mask_dir = output_dir / "masks"
    prob_dir = output_dir / "probabilities"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

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
    sample_index = 0

    for batch in tqdm(loader, desc=f"export {model_name}/{split}"):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(image))
        pred = (prob >= threshold).float()
        batch_dice = dice_score(pred, mask).cpu()
        batch_iou = iou_score(pred, mask).cpu()
        batch_precision = precision_score(pred, mask).cpu()
        batch_recall = recall_score(pred, mask).cpu()

        for idx, image_path in enumerate(batch["image_path"]):
            stem = f"{sample_index:05d}_{Path(image_path).stem}"
            prob_path = prob_dir / f"{stem}.npy"
            pred_path = mask_dir / f"{stem}.png"
            prob_array = prob[idx, 0].detach().cpu().numpy().astype("float32")
            pred_array = (prob_array >= threshold).astype("uint8") * 255
            np.save(prob_path, prob_array)
            Image.fromarray(pred_array).save(pred_path)
            rows.append(
                {
                    "image_path": image_path,
                    "mask_path": batch["mask_path"][idx],
                    "prediction_path": str(pred_path),
                    "probability_path": str(prob_path),
                    "dice": float(batch_dice[idx]),
                    "iou": float(batch_iou[idx]),
                    "precision": float(batch_precision[idx]),
                    "recall": float(batch_recall[idx]),
                }
            )
            sample_index += 1

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_predictions(args.params, args.model, args.split)


if __name__ == "__main__":
    main()
