from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_params(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return cwd


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
