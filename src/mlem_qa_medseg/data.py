from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SegmentationDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int, augment: bool = False) -> None:
        self.rows = pd.read_csv(csv_path).to_dict("records")
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment and torch.rand(()) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = torch.from_numpy((np.array(mask) > 127).astype("float32")).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

