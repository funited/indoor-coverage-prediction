"""
Two-leg training pipeline.

Leg 1: predict coarse heatmap from (permittivity, conductivity, FSPL).
Leg 2: predict refined heatmap from (conductivity, FSPL, upsampled coarse pred).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import tensorflow as tf
from PIL import Image


# -------- training helpers --------

def _default_callbacks(
    checkpoint_path: str | Path,
    log_dir: str | Path = "logs",
    early_stop_patience: int = 3,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path),
                                           verbose=1, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=early_stop_patience,
                                         monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


def train_leg(
    model: tf.keras.Model,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.3,
    checkpoint_path: str | Path = "checkpoint.keras",
    log_dir: str | Path = "logs",
    callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
) -> tf.keras.callbacks.History:
    """Fit a model with sensible defaults. Returns the History object."""
    cbs = list(callbacks) if callbacks is not None else _default_callbacks(
        checkpoint_path, log_dir
    )
    return model.fit(
        X, Y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cbs,
    )


# -------- coarse-prediction handling --------

def predict_coarse(
    model: tf.keras.Model,
    X: np.ndarray,
    *,
    batch_size: int = 32,
    save_dir: str | Path | None = None,
    name_prefix: str = "Coarse_Pred",
) -> np.ndarray:
    """Run leg-1 to get coarse predictions.

    Returns:
        np.ndarray of shape (N, H_out, W_out) -- in-memory, full float32 precision.

    If `save_dir` is given, predictions are also written as lossless PNGs
    (uint8) for archival/reproducibility. The in-memory return is unaffected.
    """
    preds = model.predict(X, batch_size=batch_size).squeeze(axis=-1)  # (N, H, W)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(preds):
            arr = np.clip(p, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(save_dir / f"{name_prefix}_{i}.png")
    return preds


def build_second_leg_input(
    X: np.ndarray,
    coarse: np.ndarray,
    target_hw: tuple[int, int] = (256, 256),
    keep_channels: tuple[int, int] = (1, 2),  # (conductivity, FSPL)
) -> np.ndarray:
    """Stack 2nd-leg inputs: (conductivity, FSPL, upsampled coarse pred).

    The coarse predictions from leg-1 are at half resolution; we upsample
    them with nearest-neighbor to match `target_hw`, then stack.

    NOTE: by default we drop channel 0 (permittivity) and substitute the
    coarse prediction. That matches the original two-leg design.
    """
    if X.shape[0] != coarse.shape[0]:
        raise ValueError(f"Sample counts differ: X={X.shape[0]} coarse={coarse.shape[0]}")

    H, W = target_hw
    upsampled = np.empty((coarse.shape[0], H, W), dtype=np.uint8)
    for i, p in enumerate(coarse):
        arr = np.clip(p, 0, 255).astype(np.uint8)
        upsampled[i] = np.array(Image.fromarray(arr).resize((W, H), Image.NEAREST))

    a, b = keep_channels
    return np.stack([X[..., a], X[..., b], upsampled], axis=-1)
