from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_CONFIGS = [(5, 1.0), (11, 2.0), (21, 4.0)]


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Sample the 2D Gaussian formula and normalize the finite kernel."""
    if size < 1 or size % 2 != 1 or sigma <= 0:
        raise ValueError("size must be a positive odd integer; sigma must be positive")
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    x, y = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def opencv_kernel(size: int, sigma: float) -> np.ndarray:
    """OpenCV is used here solely to make the reference kernel."""
    one_dimensional = cv2.getGaussianKernel(size, sigma)
    return one_dimensional @ one_dimensional.T


def _reflect101_indices(length: int, before: int, after: int) -> np.ndarray:
    positions = np.arange(-before, length + after)
    if length == 1:
        return np.zeros_like(positions)
    period = 2 * (length - 1)
    positions %= period
    return np.where(positions < length, positions, period - positions)


def manual_pad(
    image: np.ndarray, amount: int, mode: str = "reflect", value: int = 0
) -> np.ndarray:
    """Pad an HxW or HxWxC array without NumPy/OpenCV padding functions.

    reflect means mirror around the edge pixel (OpenCV REFLECT_101 semantics).
    """
    if amount < 0:
        raise ValueError("padding amount must be nonnegative")
    if image.ndim not in (2, 3) or not image.size:
        raise ValueError("image must be a nonempty 2D or 3D array")
    height, width = image.shape[:2]
    if mode == "constant":
        shape = (height + 2 * amount, width + 2 * amount) + image.shape[2:]
        padded = np.full(shape, value, dtype=image.dtype)
        padded[amount : amount + height, amount : amount + width] = image
        return padded
    if mode == "reflect":
        rows = _reflect101_indices(height, amount, amount)
        columns = _reflect101_indices(width, amount, amount)
        return image[rows[:, None], columns[None, :]]
    raise ValueError("mode must be 'constant' or 'reflect'")


def manual_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute each shifted, weighted image slice; no library filter call."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("kernel must be a square array with odd width")
    radius = kernel.shape[0] // 2
    source = image.astype(np.float64)
    padded = manual_pad(source, radius, "reflect")
    height, width = image.shape[:2]
    result = np.zeros_like(source)
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            result += kernel[row, column] * padded[row : row + height, column : column + width]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def parse_config(value: str) -> tuple[int, float]:
    try:
        size_text, sigma_text = value.split(":", 1)
        size, sigma = int(size_text), float(sigma_text)
        gaussian_kernel(size, sigma)
        return size, sigma
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("use SIZE:SIGMA, e.g. 11:2; SIZE must be odd") from exc


def save_kernel_chart(configs: list[tuple[int, float]], output: Path) -> None:
    fig, axes = plt.subplots(len(configs), 3, figsize=(11, 3.5 * len(configs)), squeeze=False)
    for row, (size, sigma) in enumerate(configs):
        manual = gaussian_kernel(size, sigma)
        reference = opencv_kernel(size, sigma)
        difference = np.abs(manual - reference)
        shared_max = max(manual.max(), reference.max())
        for column, (data, title) in enumerate(
            [(manual, "Manual 2D Gaussian"), (reference, "OpenCV reference"), (difference, "Absolute difference")]
        ):
            ax = axes[row, column]
            ax.imshow(data, cmap="viridis", vmin=0, vmax=shared_max)
            detail = f"max={difference.max():.1e}" if column == 2 else f"{size}x{size}, sigma={sigma:g}"
            ax.set_title(f"{title}\n{detail}")
            ax.set_xticks([])
            ax.set_yticks([])
        print(f"{size}x{size}, sigma={sigma:g}: sum={manual.sum():.12f}, max error={difference.max():.3e}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_blur_chart(image: np.ndarray, configs: list[tuple[int, float]], output: Path) -> None:
    images = [(image, "Original")]
    for size, sigma in configs:
        images.append((manual_filter(image, gaussian_kernel(size, sigma)), f"Manual blur: {size}x{size}, sigma={sigma:g}"))
    columns = 2
    rows = (len(images) + 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4.8 * rows), squeeze=False)
    for ax, (data, title) in zip(axes.flat, images):
        ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_padding_chart(image: np.ndarray, amount: int, output: Path) -> None:
    images = [
        (image, "Original"),
        (manual_pad(image, amount, "constant", 0), f"Constant zero, {amount} px"),
        (manual_pad(image, amount, "reflect"), f"Reflect 101, {amount} px"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (data, title) in zip(axes, images):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(__file__).with_name("photo_city.jpg"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--config", type=parse_config, action="append", help="repeat SIZE:SIGMA, e.g. --config 5:1")
    parser.add_argument("--padding", type=int, default=32, help="border width in pixels")
    args = parser.parse_args()
    configs = args.config or DEFAULT_CONFIGS
    image = np.asarray(Image.open(args.image).convert("RGB"))
    args.output.mkdir(parents=True, exist_ok=True)
    save_kernel_chart(configs, args.output / "01_gaussian_kernels.png")
    save_blur_chart(image, configs, args.output / "02_manual_blur.png")
    save_padding_chart(image, args.padding, args.output / "03_manual_padding.png")
    print(f"Saved results in {args.output}")


if __name__ == "__main__":
    main()
