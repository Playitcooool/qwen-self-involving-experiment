from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import random
from pathlib import Path
from typing import Iterable

import torch


PACKAGES = ("torch", "transformers", "peft", "accelerate", "pandas", "pyarrow", "numpy", "safetensors")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: object) -> str:
    def canonical(item: object):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot canonically serialize {type(item).__name__}")

    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def package_versions(packages: Iterable[str] = PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def directory_fingerprint(path: Path) -> dict[str, object]:
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append({"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    return {"path": str(path.resolve()), "files": files, "aggregate_sha256": content_sha256(files)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
