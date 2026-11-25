import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path=input("请输入想要处理图片的路径：")
img=cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("图片读取失败，请检查路径是否正确")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=img.astype(float)
n=int(input("你想对图片使用几个滤波核处理："))
print(f"请输入{n}组核大小（奇数）和sigma，每组占一行：")
kers=[]
outs=[]
sizes=[]
sigmas=[]
for i in range(n):
    size_str,sigma_str=input().split()
    sizes.append(int(size_str))
    sigmas.append(float(sigma_str))

#滤波不能用库函数，生成滤波核应该可以罢:)
def mkkukernel(size,sigma):
    kernel_ver=cv2.getGaussianKernel(size,sigma)
    kernel=kernel_ver@kernel_ver.T
    return kernel
for i in range(n):
    kers.append(mkkukernel(sizes[i],sigmas[i]))

for i in range(n):
    print(f"开始用第{i+1}个滤波核对图像进行处理")
    h,w,chanel=img.shape
    kh,kw=kers[i].shape
    output=np.zeros(((h-kh+1),(w-kw+1),chanel))
    for j in range(h-kh+1):
        for k in range(w-kw+1):
            for c in range(chanel):
                output[j,k,c]=np.sum(img[j:j+kh,k:k+kw,c]*kers[i])
    # windows=np.lib.stride_tricks.sliding_window_view(img,(kh,kw),axis=(0,1))
    # output=np.sum(windows*kers[i],axis=(-2,-1))
    output=np.clip(output,0,255).astype(np.uint8)
    outs.append(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
    print(f"第{i+1}个滤波核处理完成！")

plt.figure(figsize=(10*(n+1),7))
plt.subplot(1,n+1,1)
plt.imshow(img_rgb)
plt.title("原图")
plt.axis("off")
for i in range(n):
    plt.subplot(1,n+1,i+2)
    plt.imshow(outs[i])
    plt.title(f"经过第{i+1}个滤波核(Size={sizes[i]},sigma={sigmas[i]})处理后的图像")
    plt.axis("off")
plt.tight_layout()
plt.show()


    