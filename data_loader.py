"""
Data loading for the Indoor Coverage Prediction dataset.

Dataset: https://huggingface.co/datasets/funited/Indoor_Coverage_Prediction

For each (band, split), the model input X is built by stacking three channels:
    channel 0: Mixed_Permittivity_channel   (per-sample)
    channel 1: Mixed_Conductivity_channel   (per-sample)
    channel 2: FSPL                         (3 images, each broadcast across N/3 samples)

The target Y is the corresponding Heatmap_mixed image, downsampled 2x.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

REPO_ID = "funited/Indoor_Coverage_Prediction"
IMG_WIDTH = 256
IMG_HEIGHT = 256


def download_dataset(band: str, split: str, cache_dir: str | None = None) -> Path:
    """Download (or reuse cached) dataset folders for a given band/split.

    Args:
        band:      '5GHz' or '28GHz'
        split:     'Low' or 'High'
        cache_dir: optional local dir; if None, uses Hugging Face's default cache.

    Returns:
        Path to the local directory containing the requested band/split.
    """
    local_root = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{band}/{split}/**"],
        local_dir=cache_dir,
    )
    return Path(local_root) / band / split


def _load_folder(folder: Path, ext: str, size=(IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    """Load all images with the given extension from `folder`, sorted by filename.

    All images are converted to grayscale ('L' mode) so output is (N, H, W).
    """
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == f".{ext.lower()}")
    if not paths:
        raise FileNotFoundError(f"No .{ext} images found in {folder}")
    return np.stack(
        [np.array(Image.open(p).convert("L").resize(size)) for p in paths],
        axis=0,
    )


def load_data(
    band: str = "5GHz",
    split: str = "High",
    train_size: int = 2900,
    seed: int = 42,
    cache_dir: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_train, Y_train, X_test, Y_test) for the given band/split.

    X shape: (N, IMG_HEIGHT, IMG_WIDTH, 3)   -- 3 stacked grayscale channels
    Y shape: (N, IMG_HEIGHT // 2, IMG_WIDTH // 2)
    """
    base = download_dataset(band, split, cache_dir=cache_dir)

    permittivity = _load_folder(base / "Mixed_Permittivity_channel", "jpg")
    conductivity = _load_folder(base / "Mixed_Conductivity_channel", "jpg")
    fspl         = _load_folder(base / "FSPL", "jpg")
    heatmap      = _load_folder(
        base / "Heatmap_mixed", "png",
        size=(IMG_WIDTH // 2, IMG_HEIGHT // 2),
    )

    # Sanity check: FSPL is broadcast across equal-sized groups
    n_samples, n_fspl = len(permittivity), len(fspl)
    if n_samples != len(conductivity) or n_samples != len(heatmap):
        raise ValueError(
            f"Channel size mismatch: permittivity={n_samples}, "
            f"conductivity={len(conductivity)}, heatmap={len(heatmap)}"
        )
    if n_samples % n_fspl != 0:
        raise ValueError(f"{n_samples} samples not evenly divisible by {n_fspl} FSPL images")
    per_group = n_samples // n_fspl

    # Stack the 3 grayscale channels into (N, H, W, 3); each FSPL image repeats `per_group` times.
    X = np.stack(
        [permittivity, conductivity, np.repeat(fspl, per_group, axis=0)],
        axis=-1,
    )
    Y = heatmap

    # Reproducible shuffle (the *correct* way to seed numpy)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    X, Y = X[idx], Y[idx]

    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data(band="5GHz", split="High")
    print(f"X_train: {X_train.shape}    Y_train: {Y_train.shape}")
    print(f"X_test:  {X_test.shape}    Y_test:  {Y_test.shape}")
