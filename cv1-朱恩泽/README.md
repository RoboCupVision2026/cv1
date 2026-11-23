# CV1 作业

作业中的`photo_city.jpg` 和 `photo_monument.jpg` 是两张实拍照片缩小得到的，两张照片都用于生成结果，运行时可通过 `--image` 切换

## 环境

Ubuntu 下需要 Python 3、NumPy、OpenCV、Matplotlib、Pillow

```bash
sudo apt update
sudo apt install python3-numpy python3-opencv python3-matplotlib python3-pil
python3 homework.py
python3 homework.py --image photo_monument.jpg --output results/monument
```

如果另装了不兼容的 NumPy，系统 apt 安装的库可能被覆盖。使用了以下命令成功运行：

```bash
PYTHONPATH=/usr/lib/python3/dist-packages MPLCONFIGDIR=/tmp/cv1-matplotlib python3 homework.py
PYTHONPATH=/usr/lib/python3/dist-packages MPLCONFIGDIR=/tmp/cv1-matplotlib python3 homework.py --image photo_monument.jpg --output results/monument
```

可自行设置任意正奇数核大小、正数 σ，重复 `--config` 至少三次，并指定输入图、输出目录和填充宽度：

```bash
python3 homework.py --config 3:0.8 --config 9:1.5 --config 17:3 --image photo_city.jpg --output results --padding 32
```

## 结果

二维高斯核按 $G(x,y)=\exp[-(x^2+y^2)/(2\sigma^2)]$ 在离散坐标上采样，再除以所有元素之和归一化。OpenCV 仅用 `getGaussianKernel` 生成一维参考核，外积形成二维参考。`01_gaussian_kernels.png` 展示的是三组参数的手写核、OpenCV 核和绝对误差。 `manual_filter` 对每个核位置计算移位图像与权重的乘积，再累加得到输出。边界使用手写的镜像填充。滤波过程没有调用 OpenCV 滤波函数。`results/02_manual_blur.png` 和 `results/monument/02_manual_blur.png` 分别展示两张原图与三组手写高斯模糊，核/σ 越大通常模糊越明显。

`manual_pad` 实现零值常量填充及 Reflect 101 镜像填充，均未调用 OpenCV 或 NumPy 的填充函数。`03_manual_padding.png` 展示两种边界处理的图像结果。

默认三组参数 (5,1)、(11,2)、(21,4) 的手写核与 OpenCV 核最大元素误差均在浮点舍入量级。结果图保存在 `results/`。
