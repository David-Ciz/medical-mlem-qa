from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import resolve_path


def analyze(model_a: str, model_b: str, split: str, metric: str) -> None:
    path_a = resolve_path("predictions") / model_a / split / "manifest.csv"
    path_b = resolve_path("predictions") / model_b / split / "manifest.csv"
    data_a = pd.read_csv(path_a)
    data_b = pd.read_csv(path_b)
    merged = data_a.merge(
        data_b,
        on=["image_path", "mask_path"],
        suffixes=(f"_{model_a}", f"_{model_b}"),
    )
    metric_a = f"{metric}_{model_a}"
    metric_b = f"{metric}_{model_b}"
    a_wins = merged[metric_a] > merged[metric_b]
    b_wins = merged[metric_b] > merged[metric_a]
    ties = merged[metric_a] == merged[metric_b]
    oracle = merged[[metric_a, metric_b]].max(axis=1)
    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "split": split,
        "metric": metric,
        "n": int(len(merged)),
        "model_a_mean": float(merged[metric_a].mean()),
        "model_b_mean": float(merged[metric_b].mean()),
        "model_a_wins": int(a_wins.sum()),
        "model_b_wins": int(b_wins.sum()),
        "ties": int(ties.sum()),
        "model_a_win_fraction": float(a_wins.mean()),
        "model_b_win_fraction": float(b_wins.mean()),
        "oracle_top1_mean": float(oracle.mean()),
        "oracle_gain_over_best_single": float(oracle.mean() - max(merged[metric_a].mean(), merged[metric_b].mean())),
    }
    output_dir = resolve_path("reports") / "complementarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_{model_a}_vs_{model_b}_{metric}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="unet")
    parser.add_argument("--model-b", default="deeplabv3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--metric", default="dice")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.model_a, args.model_b, args.split, args.metric)


if __name__ == "__main__":
    main()

