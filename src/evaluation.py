"""Evaluation metrics and visualization helpers."""
from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def overall_rmse_mae(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Flattened RMSE and MAE over all pixels of all samples."""
    p, a = predictions.flatten(), actuals.flatten()
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    mae  = float(np.mean(np.abs(p - a)))
    return {"rmse": rmse, "mae": mae}


def _resize_to(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    return zoom(arr.squeeze(), (H / arr.shape[-2], W / arr.shape[-1]))


def plot_prediction_vs_truth(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    display_hw: tuple[int, int] = (65, 115),     # original floor-plan aspect
    cmap: str = "jet",
    dbm_range: tuple[float, float] = (-125.5, -54.6),
    titles: tuple[str, str] = ("Prediction", "RT Simulation"),
):
    """Side-by-side heatmap comparison with a shared colorbar (dBm)."""
    pred = _resize_to(prediction, display_hw)
    truth = _resize_to(ground_truth, display_hw)

    vmin = min(pred.min(), truth.min())
    vmax = max(pred.max(), truth.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, title in zip(axes, (pred, truth), titles):
        ax.imshow(img, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    cbar = fig.colorbar(axes[1].images[0], ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("Signal Strength (dBm)")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{dbm_range[0]} dBm", f"{dbm_range[1]} dBm"])
    plt.show()


def plot_error_cdf(prediction: np.ndarray, ground_truth: np.ndarray,
                   *, bins: int = 256, value_range: tuple[float, float] = (0, 256)):
    """CDF of |prediction - ground_truth| pixel differences."""
    diff = np.abs(prediction.squeeze() - ground_truth.squeeze()).flatten()
    hist, edges = np.histogram(diff, bins=bins, range=value_range, density=True)
    cdf = np.cumsum(hist) * np.diff(edges)
    plt.plot(cdf, color="blue")
    plt.title("CDF of Image Difference")
    plt.xlabel("Pixel Intensity Difference")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.show()


def plot_training_curves(history, *, threshold: Optional[float] = None):
    """Loss curves with an optional cap to keep early-epoch spikes from
    crushing the y-axis."""
    train = history.history["loss"]
    val   = history.history["val_loss"]
    if threshold is not None:
        train = [min(t, threshold) for t in train]
        val   = [min(v, threshold) for v in val]
    epochs = range(len(train))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train, "g-", label="Training error")
    plt.plot(epochs, val,   "r-", label="Validation error")
    plt.title("Training and Validation Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error (MSE)")
    plt.legend()
    plt.show()
    print(f"Final training error:   {train[-1]:.4f}")
    print(f"Final validation error: {val[-1]:.4f}")
