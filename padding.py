import numpy as np
import matplotlib.pyplot as plt
import cv2


def zero_padding(image, pad):
    """给二维灰度图的四周填充 0"""

    height, width = image.shape

    result = np.zeros((height + 2 * pad, width + 2 * pad),dtype=image.dtype)

    result[pad:pad + height, pad:pad + width] = image

    return result


def copy_padding(image, pad):
    """使用最近的边缘像素复制填充"""

    result = zero_padding(image, pad)

    height, width = image.shape
    output_height, output_width = result.shape

    for row in range(output_height):
        
        source_row = min(max(row - pad, 0), height - 1)

        for col in range(output_width):

            source_col = min(max(col - pad, 0), width - 1)
            result[row, col] = image[source_row, source_col]

    return result


def reflect_index(index, length):
    """将越界下标镜像映射"""

    if length == 1:
        return 0

    # 填充很宽时，可能需要多次反射
    while index < 0 or index >= length:

        if index < 0:
            index = -index
        else:
            # 越过右边界或下边界
            index = 2*(length-1) - index

    return index


def reflection_padding(image, pad):
    """通过镜像反射原图的像素，填充四周。"""

    result = zero_padding(image, pad)

    height, width = image.shape
    output_height, output_width = result.shape

    for row in range(output_height):

        source_row = reflect_index(row - pad, height)

        for col in range(output_width):

            source_col = reflect_index(col - pad, width)
            result[row, col] = image[source_row, source_col]

    return result


if __name__ == "__main__":

    image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

    height, width = image.shape

    top = max((height-300) // 2, 0)
    left = max((width-300) // 2, 0)

    image = image[top:top + 300, left:left + 300]

    pad = 30

    zero = zero_padding(image, pad)
    copied = copy_padding(image, pad)
    reflected = reflection_padding(image, pad)

    pictures = [
        ("Original", image),
        ("Zero padding", zero),
        ("Copy padding", copied),
        ("Reflection padding", reflected)]

    plt.figure(figsize=(16, 4))

    for index in range(len(pictures)):
        name, picture = pictures[index]

        print(f"{name}: {picture.shape}")

        plt.subplot(1, len(pictures), index + 1)
        plt.imshow(picture, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name}\n{picture.shape}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("padding_comparison.png", dpi=150)
    plt.show()