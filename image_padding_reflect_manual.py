from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


INPUT_PATH = Path(__file__).with_name("test3.jpg")
OUTPUT_PATH = Path(__file__).with_name("output") / "padded_reflect_test3.jpg"
PADDING = 300


def reflect_indices(length: int, padding: int) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be at least 1")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    indices = np.empty(length + 2 * padding, dtype=np.int64)

    if length == 1:
        indices[:] = 0
        return indices

    period = 2 * length - 2
    for index in range(-padding, length + padding):
        position = index % period
        if position >= length - 1:
            position = period - position
        indices[index + padding] = position

    return indices


def reflect_across_edge(image: np.ndarray, padding: int) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D NumPy array")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    height, width = image.shape[:2]
    row_indices = reflect_indices(height, padding)
    column_indices = reflect_indices(width, padding)

    padded = np.empty(
        (height + 2 * padding, width + 2 * padding) + image.shape[2:],
        dtype=image.dtype,
    )

    # 通过坐标映射逐像素填充，实现不包含边缘的 reflect 方式。
    for output_row, input_row in enumerate(row_indices):
        for output_column, input_column in enumerate(column_indices):
            padded[output_row, output_column] = image[input_row, input_column]

    return padded


def main() -> None:
    image=cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    padded_image = reflect_across_edge(image, PADDING)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    padded_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original ({image.shape[1]} x {image.shape[0]})")
    axes[1].imshow(padded_rgb)
    axes[1].set_title(
        f"Padded ({padded_image.shape[1]} x {padded_image.shape[0]})"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Manual reflect across edge, padding = {PADDING}px")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
