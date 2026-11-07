import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

def getKe(size,sigma):
    sig2=sigma*sigma*2.0
    center=(size//2)
    Sum=0.0

    #参数准备

    ke=np.zeros((size,size),dtype=np.float64)
    for i in range(size):
        nu_x=i-center
        for j in range(size):
            nu_y=j-center
            ke[i][j]=math.exp(-(nu_x*nu_x+nu_y*nu_y)/sig2)
            Sum+=ke[i][j]

    #计算核数组

    for i in range(size):
        nu_x=i-center
        for j in range(size):
            ke[i][j]=ke[i][j]/Sum
    return ke

#归一化

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

case=np.zeros(3)
case=[9,21,41]
sgm=[2.0,3.0,4.0]
for i in range(3):
    ax = axes[i]
    sz=case[i]
    sg=sgm[i]
    K=getKe(case[i],sg)
    im = ax.imshow(K,cmap="viridis",interpolation="nearest")
    ax.set_title("size=%d, sigma=%.1f"%(sz,sg))
    fig.colorbar(im,ax=ax)
    kcv = cv2.getGaussianKernel(sz, sg)
    kcv = np.outer(kcv, kcv)
    #获得cv2的核
    print("size=%d，sigma=%.1f，与cv2最大差=%.2e" % (sz, sg, np.abs(K - kcv).max()))

fig.tight_layout()
fig.savefig("核可视化.png", dpi=150)