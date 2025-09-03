from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_spectra_and_filters(lambdas: np.ndarray, M: np.ndarray, channel_names: list[str], fluor_names: list[str]) -> None:
    L, K = M.shape
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    # Approximate per-channel response area by normalizing rows of M for visualization
    for k in range(K):
        ax.plot(range(L), M[:, k], label=f"{fluor_names[k]}")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Integral response")
    ax.legend()
    ax.set_title("System matrix M (rows=channels, cols=fluors)")
    plt.tight_layout()


def quicklook_abundances(A: np.ndarray, H: int, W: int, titles: Optional[list[str]] = None) -> None:
    K, P = A.shape
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.imshow(A[k].reshape(H, W), cmap="magma")
        t = titles[k] if titles and k < len(titles) else f"k={k}"
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()


