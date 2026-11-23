from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("output") / "gaussian_kernel_comparison.png"
KERNEL_SIZE = 11
SIGMA = 3.0


def manual_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    # 核大小必须为正奇数，以保证存在唯一的中心像素。
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    # 高斯分布的标准差必须大于零。
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # 以核中心为原点，计算从 -radius 到 radius 的坐标范围。
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    y = np.arange(-radius, radius + 1, dtype=np.float64)
    # meshgrid 将一维坐标扩展为二维坐标矩阵，xx 和 yy 分别表示横纵坐标。
    xx, yy = np.meshgrid(x, y)

    # 使用二维高斯公式 G(x,y)=exp(-(x^2+y^2)/(2σ^2)) 逐点计算权重。
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    # 除以全部权重之和，使核的权重总和为 1，滤波不会改变图像整体亮度。
    kernel /= np.sum(kernel)
    return kernel


def opencv_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    # 参数校验与手动实现保持一致，确保两种方法使用相同的核尺寸和方差。
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # 先让 OpenCV 生成一维高斯核，并使用双精度浮点数减少计算误差。
    one_dimensional = cv2.getGaussianKernel(kernel_size, sigma, ktype=cv2.CV_64F)
    # 二维高斯核等于一维核与自身的转置矩阵相乘。
    return one_dimensional @ one_dimensional.T



def visualize_kernels(
    manual_kernel: np.ndarray,
    opencv_kernel: np.ndarray,
    kernel_size: int,
    sigma: float,
) -> None:
    # 计算差异矩阵，以便在第三幅子图中展示误差分布。
    difference = manual_kernel - opencv_kernel
    # 设置坐标范围，使核中心位于坐标原点。
    kernel_extent = (-(kernel_size // 2), kernel_size // 2,
                     -(kernel_size // 2), kernel_size // 2)

    # 创建一行三列画布，分别展示手动核、OpenCV 核和误差。
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = (
        (manual_kernel, "Manual 2D Gaussian"),
        (opencv_kernel, "OpenCV getGaussianKernel"),
        (difference, "Absolute Difference"),
    )

    # 遍历三个子图，为每个矩阵设置颜色映射、坐标轴和标题。
    for axis, (kernel, title) in zip(axes, images):
        # 差异图使用红蓝色映射；核权重图使用渐变色映射。
        color_map = "RdBu_r" if title == "Absolute Difference" else "viridis"
        # 差异图显示绝对误差，避免正负误差互相抵消。
        shown = np.abs(kernel) if title == "Absolute Difference" else kernel
        image = axis.imshow(
            shown,
            extent=kernel_extent,
            origin="lower",
            cmap=color_map,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        # 显示与核坐标相对应的刻度。
        axis.set_xticks(np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
        axis.set_yticks(np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
        # 为每幅子图添加颜色条。
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    # 添加整体标题，并自动调整子图间距。
    figure.suptitle(
        f"2D Gaussian Kernel Comparison (size={kernel_size}, sigma={sigma})"
    )
    figure.tight_layout()
    # 保存可视化结果，随后调用 show 在桌面环境中展示。
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:

    # 分别调用手动公式和 OpenCV 方法生成相同配置的二维核。
    manual_kernel = manual_gaussian_kernel(KERNEL_SIZE, SIGMA)
    opencv_kernel = opencv_gaussian_kernel(KERNEL_SIZE, SIGMA)

    # 使用 Matplotlib 展示核权重及差异。
    visualize_kernels(
        manual_kernel,
        opencv_kernel,
        KERNEL_SIZE,
        SIGMA,
    )

if __name__ == "__main__":
    main()
