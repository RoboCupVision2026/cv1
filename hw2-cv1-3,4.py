import matplotlib.pyplot as plt
import math
from pathlib import Path
import numpy as np
from PIL import Image


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

    return np.array(GS)


def blur(image, kernel, size, mode):
    image = np.array(image, dtype=np.float64)
    r = size // 2
    result = np.zeros_like(image, dtype=np.float64)[r:-r, r:-r, :]

    for i in range(size):
        for j in range(size):
            region = image[i : i + result.shape[0],j : j + result.shape[1],:,]
            result += kernel[i, j] * region

    padded = np.pad(result, ((r, r), (r, r), (0, 0)),mode)

    return np.rint(padded).astype(np.uint8)


def main():
    image_path = input("请输入原图路径：").strip()
    image_path = Path(image_path).expanduser()

    with Image.open(image_path) as source:
        original = np.asarray(source.convert("RGB"), dtype=np.uint8)
    size, sigma = int(input("大小（正奇数）：")), float(input("标准差："))
    kernel = gaosi(size, sigma)
    m = int(input("请输入填充模式：\n1:edge\n2:constant\n3:reflect\n4:symmetric\n"))
    mode = ["edge", "constant", "reflect", "symmetric"][m - 1]

    blurred = blur(original, kernel, size, mode)

    output_dir = Path.cwd()

    figure, axes = plt.subplots(1, 2, figsize=(6, 10))
    axes = axes.ravel()
    axes[0].imshow(original)
    axes[0].set_title("原图")
    axes[0].axis("off")

    axes[1].imshow(blurred)
    axes[1].set_title(f"大小：{size}, 标准差：{sigma}")
    axes[1].axis("off")
    output_path = output_dir / f"模糊后_大小：{size}_标准差：{sigma:g}padding模式：{mode}.png"
    Image.fromarray(blurred).save(output_path)

    figure.suptitle("模糊对比")
    figure.tight_layout()
    comparison_path = output_dir / f"模糊对比_大小：{size}_标准差：{sigma:g}padding模式：{mode}.png"
    figure.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.show()


while True:
    main()