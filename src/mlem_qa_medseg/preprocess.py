from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from pathlib import Path

import certifi
import requests
from requests.exceptions import SSLError

from .config import load_params, resolve_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_file(url: str, destination: Path, allow_insecure_ssl: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}")
    try:
        response = requests.get(url, stream=True, timeout=60, verify=certifi.where())
    except SSLError:
        if not allow_insecure_ssl:
            raise
        print("Warning: SSL verification failed; retrying with verification disabled.")
        response = requests.get(url, stream=True, timeout=60, verify=False)

    with response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_zip(zip_path: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def find_dataset_root(raw_dir: Path) -> Path:
    candidates = [raw_dir, *[p for p in raw_dir.rglob("*") if p.is_dir()]]
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find Kvasir-SEG images/ and masks/ directories under {raw_dir}"
    )


def collect_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []

    for image_path in images:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            matches = list(mask_dir.glob(f"{image_path.stem}.*"))
            mask_path = matches[0] if matches else mask_path
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing.append(image_path.name)

    if missing:
        print(f"Warning: {len(missing)} images did not have a matching mask.")
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {dataset_root}")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = round(n_total * test_fraction)
    n_val = round(n_total * val_fraction)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def write_split_csv(split_name: str, pairs: list[tuple[Path, Path]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path"])
        writer.writeheader()
        for image_path, mask_path in pairs:
            writer.writerow({"image_path": str(image_path), "mask_path": str(mask_path)})


def preprocess(params_path: Path) -> None:
    params = load_params(params_path)
    data_params = params["data"]
    raw_dir = resolve_path(data_params["raw_dir"])
    processed_dir = resolve_path(data_params["processed_dir"])
    url = str(data_params.get("url", "")).strip()
    if url:
        zip_path = raw_dir / "kvasir-seg.zip"
        download_file(
            url,
            zip_path,
            allow_insecure_ssl=bool(data_params.get("allow_insecure_ssl", False)),
        )
        extract_zip(zip_path, raw_dir)
    else:
        print(f"Using mounted/local dataset under {raw_dir}")

    dataset_root = find_dataset_root(raw_dir)
    pairs = collect_pairs(dataset_root)
    splits = split_pairs(
        pairs,
        seed=int(params["seed"]),
        val_fraction=float(data_params["val_fraction"]),
        test_fraction=float(data_params["test_fraction"]),
    )

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    split_dir = processed_dir / "splits"
    for split_name, split_pairs_ in splits.items():
        write_split_csv(split_name, split_pairs_, split_dir)

    summary_path = processed_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"dataset_root={dataset_root}",
                f"total_pairs={len(pairs)}",
                *(f"{name}={len(items)}" for name, items in splits.items()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.params)


if __name__ == "__main__":
    main()
