import cv2
import matplotlib.pyplot as plt
import math


def gaosi(size, sigma):
    if size <= 0 or size % 2 == 0 or sigma <= 0:
        return False
    
    r = size // 2
    GS = [[0.0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            GS[i][j] = math.exp(-((i - r)**2 + (j - r)**2) / (2 * sigma**2))

    add = sum(sum(i) for i in GS)
    for i in range(size):
        for j in range(size):
            GS[i][j] = GS[i][j] / add

    return GS


def OpenCV_gaosi(size, sigma):
    if size <= 0 or size % 2 == 0 or sigma <= 0:
        return False

    kernel_1d = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)

    return kernel_1d @ kernel_1d.T


def visualize(GS, OpenCV_GS, size, sigma):
    dif = [[abs(GS[i][j] / OpenCV_GS[i][j] - 1) if OpenCV_GS[i][j] != 0 else abs(GS[i][j] / OpenCV_GS[i][j] - 1) for j in range(size)] for i in range(size)]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    plots = (
            (GS, "手搓版"),
            (OpenCV_GS, "OpenCV版"),
            (dif, "相对误差的绝对值"),
        )
    
    for axis, (kernel, title) in zip(axes, plots):
        image = axis.imshow(kernel, cmap="YlOrRd", interpolation="nearest")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xticks(range(size), range(-(size // 2), size // 2 + 1))
        axis.set_yticks(range(size), range(-(size // 2), size // 2 + 1))

        threshold = max(max(row) for row in kernel) * 0.55
        for i in range(size):
            for j in range(size):
                value = kernel[i][j]
                color = "white" if value > threshold else "black"
                label = f"{value:.2e}" if axis is axes[2] else f"{value:.4f}"
                axis.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                )

    figure.suptitle(f"高斯滤波核(大小：{size}, 标准差：{sigma})")
    output_path = f"高斯滤波核(大小：{size}, 标准差：{sigma}).png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    size, sigma = int(input("大小（正奇数）：")), float(input("标准差："))

    GS = gaosi(size, sigma)
    OpenCV_GS = OpenCV_gaosi(size, sigma)
    
    if GS:
        visualize(GS, OpenCV_GS, size, sigma)
    else:
        print("输入参数无效")


while True:
    main()