from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path(__file__).with_name("test3.jpg")
OUTPUT_PATH = Path(__file__).with_name("output") / "padded_manual_test3.jpg"
PADDING = 300


def replicate_pad(image: np.ndarray, padding: int) -> np.ndarray:
    height, width, channels = image.shape

    # 先创建一个更大的空数组，原图四周各留出 padding 像素的空间。
    padded = np.empty(
        (height + 2 * padding, width + 2 * padding, channels),
        dtype=image.dtype,
    )

    # 将原图放到大数组的中心位置。
    padded[padding : padding + height, padding : padding + width] = image

    # 复制原图第一行到上边界，复制最后一行到下边界。
    padded[:padding, padding : padding + width] = image[0:1, :, :]
    padded[padding + height :, padding : padding + width] = image[-1:, :, :]

    # 复制当前左右边界列到左、右边界。
    padded[:, :padding] = padded[:, padding : padding + 1]
    padded[:, padding + width :] = padded[:, padding + width - 1 : padding + width]

    return padded


def main() -> None:
    image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    padded_image = replicate_pad(image, PADDING)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Matplotlib 使用 RGB 顺序显示图片，因此先将 BGR 转换为 RGB。
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    padded_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    # 创建一行两列的画布，左边显示原图，右边显示处理结果。
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original ({image.shape[1]} x {image.shape[0]})")
    axes[1].imshow(padded_rgb)
    axes[1].set_title(
        f"Padded ({padded_image.shape[1]} x {padded_image.shape[0]})"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Manual edge replication, padding = {PADDING}px")
    figure.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
