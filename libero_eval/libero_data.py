"""Access to the LIBERO demonstration dataset (HuggingFaceVLA/libero, LeRobot v3 parquet layout).

Files are fetched on demand into the Hugging Face cache. The revision is pinned so that the
parquet listing, and therefore the sampled library, is identical to the one used in the paper.
"""
from __future__ import annotations
import io

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

DATASET = "HuggingFaceVLA/libero"
REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"


def parquet_files() -> list[str]:
    files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    files = [f for f in files if f.startswith("data/chunk-") and f.endswith(".parquet")]
    return sorted(files, key=lambda f: f.split("/"))


def fetch(path_in_repo: str) -> str:
    return hf_hub_download(DATASET, path_in_repo, repo_type="dataset", revision=REVISION)


def library_parquets(n: int = 50, seed: int = 0) -> list[str]:
    """The n parquet shards that make up the demonstration library (random, fixed seed)."""
    files = parquet_files()
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(files), size=min(n, len(files)), replace=False))
    return [files[i] for i in chosen]


def decode_image(cell) -> np.ndarray:
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))
