import numpy as np
import cv2
import matplotlib.pyplot as plt

from gaussian_kernel import gaussian_kernel


def manual_filter(image, kernel):
    """对二维灰度图进行不加 padding 的滤波。"""

    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1

    if output_height <= 0 or output_width <= 0:
        raise ValueError("滤波核不能大于输入图片")

    result = np.zeros((output_height, output_width),dtype=float)

    for row in range(output_height):
        for col in range(output_width):

            patch = image[row:row + kernel_height,col:col + kernel_width]
            result[row, col] = np.sum(patch * kernel) #对应位置相乘

    return result


if __name__ == "__main__":

    image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE) #cv2.imread(文件路径，读取方式)

    height, width = image.shape
    top = max((height - 300) // 2, 0)
    left = max((width - 300) // 2, 0)
    image = image[top:top + 300, left:left + 300]

    cases = [
        (5, 1.0),
        (5, 4.0),
        (9, 4.0)]

    plt.figure(figsize=(14, 4))


    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Input\n{image.shape}")
 

    for index in range(len(cases)):
        size, sigma = cases[index]

        kernel = gaussian_kernel(size, sigma)
        blurred = manual_filter(image, kernel)

        print(
            f"size={size}, sigma={sigma}, "
            f"输入尺寸={image.shape}, 输出尺寸={blurred.shape}"
        )

        plt.subplot(1, 4, index + 2) #行数 列数 位置
        plt.imshow(blurred, cmap="gray", vmin=0, vmax=255) #要显示的二维数组，颜色映射
        plt.title(f"{size}x{size}, sigma={sigma}\n{blurred.shape}")
        

    plt.tight_layout()
    plt.savefig("filter_comparison.png", dpi=150)
    plt.show()