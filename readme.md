# 图像处理基础作业：高斯滤波与 Padding 边缘填充实现

本项目为视觉组学习任务（homework2），主要基于 NumPy 从底层手写实现了图像的高斯滤波（Gaussian Blur）以及多种模式的边缘填充（Padding），并使用 Matplotlib 进行了可视化对比。

##  环境依赖

运行本项目需要安装以下 Python 库：
```bash
pip install numpy opencv-python matplotlib
```
## 文件结构说明

    mygo.py: 手写的高斯滤波类 my_GaussianBlur。包含生成二维高斯核、自定义卷积操作。

    test.py: 用于测试手写的高斯滤波，并对比opencv滤波核的效果。

    visual_show.py: 用3D图和热力图两种可视化手写的高斯滤波核，调整了三组不同的size与sigma，分别保存至：
        .\gaussian_ksize5_sigma2.5.png
        .\gaussian_ksize7_sigma1.5.png
        .\gaussian_ksize7_sigma2.5.png

    multiple_comparison.py: 高斯滤波对比脚本。生成 4 种不同 kernel size 与 4 种不同 sigma 组合的 4x4 对比图,图片保存至：
        .\multiple_comparison.png

    My_padding.py: 手写的边缘填充类 MyPadding。支持多种边缘填充模式。

    test2.py: Padding 对比脚本。生成 4 种不同填充模式（Zero, Wrap, Copy, Reflect）的横向对比图。

## 核心功能与实现思路
### 1. 高斯滤波 (Gaussian Blur)

    高斯核生成：利用 NumPy 的 meshgrid 和广播机制生成二维高斯分布，并进行归一化处理。

    卷积操作：通过双重 for 循环遍历图像的像素，利用切片获取局部图像块，与高斯核进行点乘求和。

    彩色图支持：通过 np.newaxis 将二维高斯核扩展为 (k, k, 1)，利用 NumPy 的自动广播机制，实现对 RGB 三个通道的同时卷积，且不改变通道间的相对关系。

### 2. 边缘填充 (Padding)

为了实现底层逻辑，未使用 np.pad，而是手动创建全零矩阵，并通过 NumPy 切片赋值来实现：

    Zero：边缘全部填充为 0。

    Wrap around：将图像对侧的像素复制到边缘。

    Copy edge：将图像最边缘的像素向外延伸复制（利用 NumPy 自动广播特性）。

    Reflect across edge：以边缘像素为对称轴进行镜像反射。

## 补充说明

    1.各个文件中引用的opencv库（除test.py中的cv2.GaussianBlur，这是用于对比的）均只用于图像的读取与保存（不太会plt的图像色彩转换），没有直接调任何opencv的图像处理函数，其余所有方法均由numpy实现。

    2.readme部分基本由AI生成，代码部分基本是手打的（除了padding那部分大范围基本重复的代码，我只手打了每一部分的前几行），但要用到哪些库的哪些函数，以及函数用法与用例都是AI给的。


