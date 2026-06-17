from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

from .config import resolve_path
from .metrics import dice_score


def _prediction_features(probability: np.ndarray, prefix: str) -> dict[str, float]:
    hard = probability >= 0.5
    entropy = -(
        probability * np.log(probability + 1e-8)
        + (1.0 - probability) * np.log(1.0 - probability + 1e-8)
    )
    grad_y, grad_x = np.gradient(probability)
    return {
        f"{prefix}_mean": float(probability.mean()),
        f"{prefix}_std": float(probability.std()),
        f"{prefix}_min": float(probability.min()),
        f"{prefix}_max": float(probability.max()),
        f"{prefix}_q10": float(np.quantile(probability, 0.10)),
        f"{prefix}_q25": float(np.quantile(probability, 0.25)),
        f"{prefix}_q50": float(np.quantile(probability, 0.50)),
        f"{prefix}_q75": float(np.quantile(probability, 0.75)),
        f"{prefix}_q90": float(np.quantile(probability, 0.90)),
        f"{prefix}_hard_area": float(hard.mean()),
        f"{prefix}_uncertainty": float((1.0 - np.abs(2.0 * probability - 1.0)).mean()),
        f"{prefix}_entropy": float(entropy.mean()),
        f"{prefix}_edge_strength": float((np.abs(grad_x) + np.abs(grad_y)).mean()),
    }


def _load_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return (np.array(mask) > 127).astype("float32")


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    import torch

    pred_tensor = torch.from_numpy(prediction[None, None].astype("float32"))
    target_tensor = torch.from_numpy(target[None, None].astype("float32"))
    return float(dice_score(pred_tensor, target_tensor)[0])


def _load_merged_manifests(models: list[str], split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in models:
        path = resolve_path("predictions") / model / split / "manifest.csv"
        data = pd.read_csv(path)
        data = data[
            [
                "image_path",
                "mask_path",
                "probability_path",
                "dice",
                "iou",
                "precision",
                "recall",
            ]
        ].rename(
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


def _make_features(data: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in data.to_dict("records"):
        probabilities = {
            model: np.load(row[f"probability_path_{model}"]).astype("float32") for model in models
        }
        features: dict[str, float] = {}
        for model, probability in probabilities.items():
            features.update(_prediction_features(probability, model))

        for model_a, model_b in combinations(models, 2):
            prob_a = probabilities[model_a]
            prob_b = probabilities[model_b]
            hard_a = prob_a >= 0.5
            hard_b = prob_b >= 0.5
            features[f"{model_a}_vs_{model_b}_prob_absdiff"] = float(np.abs(prob_a - prob_b).mean())
            features[f"{model_a}_vs_{model_b}_hard_disagreement"] = float((hard_a != hard_b).mean())
            features[f"{model_a}_vs_{model_b}_area_delta"] = float(hard_a.mean() - hard_b.mean())
            features[f"{model_a}_vs_{model_b}_mean_delta"] = float(prob_a.mean() - prob_b.mean())

        rows.append(features)
    return pd.DataFrame(rows)


def _score_qa_combinations(
    test_data: pd.DataFrame,
    models: list[str],
    predicted_quality: np.ndarray,
    margin: float,
    threshold: float,
) -> dict[str, object]:
    top1_indices = predicted_quality.argmax(axis=1)
    top1_rows: list[float] = []
    weighted_all_rows: list[float] = []
    within_margin_rows: list[float] = []
    above_threshold_rows: list[float] = []
    within_margin_sizes: list[int] = []
    above_threshold_sizes: list[int] = []

    for row_index, row in enumerate(test_data.to_dict("records")):
        probabilities = [
            np.load(row[f"probability_path_{model}"]).astype("float32") for model in models
        ]
        target = _load_mask(row["mask_path"], probabilities[0].shape)
        quality = predicted_quality[row_index]

        top1_probability = probabilities[top1_indices[row_index]]
        top1_rows.append(_dice(top1_probability >= 0.5, target))

        clipped_quality = np.clip(quality, 1e-4, None)
        weighted_all_probability = np.average(probabilities, axis=0, weights=clipped_quality)
        weighted_all_rows.append(_dice(weighted_all_probability >= 0.5, target))

        best_quality = float(quality.max())
        margin_indices = np.flatnonzero(quality >= best_quality - margin)
        within_margin_sizes.append(int(len(margin_indices)))
        margin_probability = np.mean([probabilities[index] for index in margin_indices], axis=0)
        within_margin_rows.append(_dice(margin_probability >= 0.5, target))

        threshold_indices = np.flatnonzero(quality >= threshold)
        if len(threshold_indices) == 0:
            threshold_indices = np.array([top1_indices[row_index]])
        above_threshold_sizes.append(int(len(threshold_indices)))
        threshold_probability = np.mean([probabilities[index] for index in threshold_indices], axis=0)
        above_threshold_rows.append(_dice(threshold_probability >= 0.5, target))

    return {
        "qa_top1_dice": float(np.mean(top1_rows)),
        "qa_weighted_all_dice": float(np.mean(weighted_all_rows)),
        "qa_within_margin_dice": float(np.mean(within_margin_rows)),
        "qa_above_threshold_dice": float(np.mean(above_threshold_rows)),
        "qa_within_margin_mean_models": float(np.mean(within_margin_sizes)),
        "qa_above_threshold_mean_models": float(np.mean(above_threshold_sizes)),
    }


def run_router(
    models: list[str],
    train_split: str,
    test_split: str,
    seed: int,
    margin: float,
    threshold: float,
) -> None:
    train_data = _load_merged_manifests(models, train_split)
    test_data = _load_merged_manifests(models, test_split)
    train_x = _make_features(train_data, models)
    test_x = _make_features(test_data, models)
    train_y = train_data[[f"dice_{model}" for model in models]]

    regressor = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    regressor.fit(train_x, train_y)
    predicted_quality = regressor.predict(test_x)
    chosen_indices = predicted_quality.argmax(axis=1)
    actual_scores = test_data[[f"dice_{model}" for model in models]].to_numpy()
    oracle_indices = actual_scores.argmax(axis=1)
    routed_scores = actual_scores[np.arange(len(test_data)), chosen_indices]
    oracle_scores = actual_scores[np.arange(len(test_data)), oracle_indices]
    model_means = {model: float(test_data[f"dice_{model}"].mean()) for model in models}
    best_single = max(model_means.values())
    combination_scores = _score_qa_combinations(
        test_data,
        models=models,
        predicted_quality=predicted_quality,
        margin=margin,
        threshold=threshold,
    )

    rows = test_data[["image_path", "mask_path"]].copy()
    for model_index, model in enumerate(models):
        rows[f"predicted_dice_{model}"] = predicted_quality[:, model_index]
        rows[f"actual_dice_{model}"] = actual_scores[:, model_index]
    rows["qa_choice"] = [models[index] for index in chosen_indices]
    rows["oracle_choice"] = [models[index] for index in oracle_indices]
    rows["qa_routed_dice"] = routed_scores
    rows["oracle_dice"] = oracle_scores

    summary = {
        "models": models,
        "train_split": train_split,
        "test_split": test_split,
        "n_train": int(len(train_data)),
        "n_test": int(len(test_data)),
        "model_dice_means": model_means,
        "best_single_dice": best_single,
        "qa_routed_dice": float(routed_scores.mean()),
        "qa_gain_over_best_single": float(routed_scores.mean() - best_single),
        "qa_combination_margin": margin,
        "qa_combination_threshold": threshold,
        **combination_scores,
        "qa_weighted_all_gain_over_best_single": float(
            combination_scores["qa_weighted_all_dice"] - best_single
        ),
        "qa_within_margin_gain_over_best_single": float(
            combination_scores["qa_within_margin_dice"] - best_single
        ),
        "qa_above_threshold_gain_over_best_single": float(
            combination_scores["qa_above_threshold_dice"] - best_single
        ),
        "oracle_top1_dice": float(oracle_scores.mean()),
        "oracle_gain_over_best_single": float(oracle_scores.mean() - best_single),
        "oracle_choice_accuracy": float((chosen_indices == oracle_indices).mean()),
        "qa_choice_counts": rows["qa_choice"].value_counts().to_dict(),
        "oracle_choice_counts": rows["oracle_choice"].value_counts().to_dict(),
    }

    output_dir = resolve_path("reports") / "qa_router"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(models)
    rows.to_csv(output_dir / f"{train_split}_to_{test_split}_{suffix}_per_image.csv", index=False)
    (output_dir / f"{train_split}_to_{test_split}_{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--train-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--margin", default=0.03, type=float)
    parser.add_argument("--threshold", default=0.80, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_router(args.models, args.train_split, args.test_split, args.seed, args.margin, args.threshold)


if __name__ == "__main__":
    main()
