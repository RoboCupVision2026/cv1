import numpy as np
import cv2
import matplotlib.pyplot as plt

n=int(input("请输入想要生成的滤波核组数："))
print(f"请输入{n}组核大小及对应sigma值，每组占一行，用空格隔开：")
size_list=[]
sigma_list=[]
ker_list=[]
for i in range(0,n):
    size_str,sigma_str=input().split()
    size_list.append(int(size_str))
    sigma_list.append(float(sigma_str))

def mkkernel(size,sigma):
    x=np.arange(-(size//2),size//2+1)
    y=np.arange(-(size//2),size//2+1)
    X,Y=np.meshgrid(x,y)
    kernel=np.exp(-(X**2+Y**2)/(2*sigma**2))
    kernel=kernel/np.sum(kernel)
    return kernel

def mkkukernel(size,sigma):
    kernel_ver=cv2.getGaussianKernel(size,sigma)
    kernel=kernel_ver@kernel_ver.T
    return kernel

for i in range(0,2*n):
    if i>=0 and i<=n-1:
        ker_list.append(mkkernel(size_list[i],sigma_list[i]))
    else:
        ker_list.append(mkkukernel(size_list[i-n],sigma_list[i-n]))

plt.figure(figsize=(4*n,8))
for i in range(1,2*n+1):
    plt.subplot(2,n,i)
    plt.imshow(ker_list[i-1],cmap="viridis")
    if i>=1 and i<=n:
        plt.title(f"手写滤波核{i}: {size_list[i-1]}*{size_list[i-1]}, sigma={sigma_list[i-1]}")
    else:
        plt.title(f"库生滤波核{i-n}: {size_list[i-n-1]}*{size_list[i-n-1]}, sigma={sigma_list[i-n-1]}")
    plt.colorbar()
plt.tight_layout()
plt.show()