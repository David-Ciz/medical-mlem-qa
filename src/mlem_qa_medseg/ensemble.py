from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import resolve_path
from .metrics import dice_score, iou_score, precision_score, recall_score


def _load_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return (np.array(mask) > 127).astype("float32")


def _score_binary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    import torch

    pred_tensor = torch.from_numpy(prediction[None, None].astype("float32"))
    target_tensor = torch.from_numpy(target[None, None].astype("float32"))
    return {
        "dice": float(dice_score(pred_tensor, target_tensor)[0]),
        "iou": float(iou_score(pred_tensor, target_tensor)[0]),
        "precision": float(precision_score(pred_tensor, target_tensor)[0]),
        "recall": float(recall_score(pred_tensor, target_tensor)[0]),
    }


def _load_manifests(models: list[str], split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in models:
        manifest_path = resolve_path("predictions") / model / split / "manifest.csv"
        data = pd.read_csv(manifest_path)
        keep = [
            "image_path",
            "mask_path",
            "probability_path",
            "dice",
            "iou",
            "precision",
            "recall",
        ]
        data = data[keep].rename(
            columns={
                "probability_path": f"probability_path_{model}",
                "dice": f"dice_{model}",
                "iou": f"iou_{model}",
                "precision": f"precision_{model}",
                "recall": f"recall_{model}",
            }
        )
        if merged is None:
            merged = data
        else:
            merged = merged.merge(data, on=["image_path", "mask_path"])
    if merged is None:
        raise ValueError("At least one model is required.")
    return merged


def evaluate_ensemble(models: list[str], split: str, threshold: float) -> None:
    manifests = _load_manifests(models, split)
    rows: list[dict[str, float | str]] = []

    for row in manifests.to_dict("records"):
        probabilities = [np.load(row[f"probability_path_{model}"]) for model in models]
        mean_probability = np.mean(probabilities, axis=0)
        target = _load_mask(row["mask_path"], mean_probability.shape)
        prediction = (mean_probability >= threshold).astype("float32")
        scores = _score_binary(prediction, target)

        model_dice = {model: float(row[f"dice_{model}"]) for model in models}
        oracle_model = max(model_dice, key=model_dice.get)
        rows.append(
            {
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "ensemble_dice": scores["dice"],
                "ensemble_iou": scores["iou"],
                "ensemble_precision": scores["precision"],
                "ensemble_recall": scores["recall"],
                "oracle_model": oracle_model,
                "oracle_dice": model_dice[oracle_model],
                **{f"dice_{model}": model_dice[model] for model in models},
            }
        )

    result = pd.DataFrame(rows)
    model_means = {model: float(result[f"dice_{model}"].mean()) for model in models}
    best_single = max(model_means.values())
    summary = {
        "models": models,
        "split": split,
        "n": int(len(result)),
        "threshold": threshold,
        "model_dice_means": model_means,
        "best_single_dice": best_single,
        "mean_probability_ensemble_dice": float(result["ensemble_dice"].mean()),
        "mean_probability_ensemble_gain_over_best_single": float(
            result["ensemble_dice"].mean() - best_single
        ),
        "oracle_top1_dice": float(result["oracle_dice"].mean()),
        "oracle_gain_over_best_single": float(result["oracle_dice"].mean() - best_single),
        "oracle_model_counts": result["oracle_model"].value_counts().to_dict(),
    }

    output_dir = resolve_path("reports") / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(models)
    result.to_csv(output_dir / f"{split}_{suffix}_per_image.csv", index=False)
    (output_dir / f"{split}_{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_ensemble(args.models, args.split, args.threshold)


if __name__ == "__main__":
    main()
