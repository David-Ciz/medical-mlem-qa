from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"

