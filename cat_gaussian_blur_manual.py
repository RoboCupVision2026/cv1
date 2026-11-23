from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gaussian_kernel_manual import manual_gaussian_kernel


INPUT_PATH = Path(__file__).with_name("cat.png")
OUTPUT_PATH = Path(__file__).with_name("output") / "cat_gaussian_blur_comparison.png"
DEFAULT_CONFIGURATIONS = (
    (3, 1.0),
    (7, 1.5),
    (11, 3.0),
    (21, 5.0),
)


def reflect_pad(image: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")

    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D NumPy array")

    # 使用 OpenCV 在上下左右四个方向进行 reflect 跨边缘镜像填充。
    return cv2.copyMakeBorder(
        image,
        radius,
        radius,
        radius,
        radius,
        borderType=cv2.BORDER_REFLECT,
    )


def manual_gaussian_filter(
    image: np.ndarray,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # 使用与二维高斯公式相同的手动归一化核。
    kernel = manual_gaussian_kernel(kernel_size, sigma)
    radius = kernel_size // 2
    image_float = image.astype(np.float64)
    original_shape = image_float.shape

    if image_float.ndim == 2:
        image_float = image_float[:, :, np.newaxis]
    elif image_float.ndim != 3:
        raise ValueError("image must be a 2D or 3D NumPy array")

    # 对图像四周进行 reflect padding，使卷积核在边缘仍有对应像素。
    padded = reflect_pad(image_float, radius)
    filtered = np.zeros(
        (original_shape[0], original_shape[1], image_float.shape[2]),
        dtype=np.float64,
    )

    # 对二维核逐元素进行加权累加；核对称，因此此处是卷积对应结果。
    for kernel_y in range(kernel_size):
        for kernel_x in range(kernel_size):
            weight = kernel[kernel_y, kernel_x]
            source = padded[
                kernel_y:kernel_y + original_shape[0],
                kernel_x:kernel_x + original_shape[1],
                :,
            ]
            filtered += weight * source

    # 四舍五入并限制像素值，再转换回 uint8。
    filtered = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
    if len(original_shape) == 2:
        return filtered[:, :, 0]
    return filtered


def main() -> None:
    configurations = DEFAULT_CONFIGURATIONS

    # 使用 OpenCV 读取 PNG，并保留 Alpha 透明度通道。
    image_bgra = cv2.imread(str(INPUT_PATH), cv2.IMREAD_UNCHANGED)
    if image_bgra is None:
        raise FileNotFoundError(f"Unable to read image: {INPUT_PATH}")
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("cat.png must contain RGBA/BGRA channels")

    # 将 OpenCV 的 BGRA 通道顺序转换为可由 Matplotlib 直接显示的 RGBA。
    alpha = image_bgra[:, :, 3]
    rgb_image = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA)[:, :, :3]

    # 保存原图和各配置的模糊结果。
    results = [("Original", rgb_image)]
    for kernel_size, sigma in configurations:
        filtered_rgb = manual_gaussian_filter(
            rgb_image,
            kernel_size,
            sigma,
        )
        results.append((
            f"{kernel_size} x {kernel_size}\nσ = {sigma}",
            filtered_rgb,
        ))

    # 为每个结果重新叠加原始透明度通道。
    rgba_results = [
        (title, np.dstack((rgb_result, alpha)))
        for title, rgb_result in results
    ]

    # 动态创建一行多列画布，原图和所有模糊结果并排展示。
    column_count = len(rgba_results)
    figure, axes = plt.subplots(
        1,
        column_count,
        figsize=(5 * column_count, 5),
    )
    if column_count == 1:
        axes = np.array([axes])

    for axis, (title, rgba_result) in zip(axes, rgba_results):
        axis.imshow(rgba_result)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle("Manual 2D Gaussian Blur Comparison")
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
