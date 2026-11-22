import numpy as np
from mygo import my_GaussianBlur
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

ksize=int(input("请输入高斯核大小："))
sigma=float(input("请输入高斯核心方差：" ))

my_blurrer=my_GaussianBlur(kernel_size=ksize,sigma=sigma)
kernel=my_blurrer.get_kernel()
title_str = f"Gaussian Kernel (ksize={ksize}, sigma={sigma})"
fig = plt.figure(num=title_str, figsize=(14, 5))

#3D图
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
k = kernel.shape[0]
x = np.arange(k)
y = np.arange(k)
X, Y = np.meshgrid(x, y)

surf = ax1.plot_surface(X, Y, kernel, cmap='viridis',
                        edgecolor='none', antialiased=True)
ax1.set_title('2D Gaussian Kernel (3D Surface)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('weight')
fig.colorbar(surf, ax=ax1, shrink=0.6, label='weight')

#热力图
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(kernel, cmap='viridis', interpolation='nearest')
ax2.set_title('2D Gaussian Kernel (Heatmap)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
for i in range(k):
    for j in range(k):
        ax2.text(j, i, f'{kernel[i, j]:.3f}',
                 ha='center', va='center', color='w', fontsize=7)
fig.colorbar(im, ax=ax2, shrink=0.8, label='weight')

plt.tight_layout()
save_name = f"gaussian_ksize{ksize}_sigma{sigma}.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()