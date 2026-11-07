import time
import numpy as np
import cv2
import math
case=[9,15,21]
def getKe(size):
    sg=0.3*((size-1)*0.5-1)+0.8
    ke=np.zeros(size,dtype=np.float64)
    sig2=2*sg*sg
    Sum=0.0
    center=size//2
    for i in range(size):
        nx=i-center
        ke[i]=math.exp(-(nx*nx)/sig2)
        Sum+=ke[i]
    for i in range(size):
        ke[i]=ke[i]/Sum
    return ke

def gauss(size,img):
    K=getKe(size)
    center=size//2
    height,width=img.shape[:2]
    Mid=np.zeros((height,width,3),dtype=np.float64)
    out=np.zeros((height,width,3),dtype=np.float64)
    for i in range(height):
        for j in range(width):
            cnt=[0,0,0]
            for k in range(size):
                nx=j+k-center
                if(nx<0): nx=0
                if(nx>width-1): nx=width-1
                for c in range(3):
                    cnt[c]+=img[i][nx][c]*K[k]
            for c in range(3):
                Mid[i][j][c]=cnt[c]
    for i in range(height):
        for j in range(width):
            cnt=[0,0,0]
            for k in range(size):
                ny=i+k-center
                if(ny<0): ny=0
                if(ny>height-1): ny=height-1
                for c in range(3):
                    cnt[c]+=Mid[ny][j][c]*K[k]
            for c in range(3):
                out[i][j][c]=cnt[c]
    return out

#高斯滤波

in_path=r"testInput.jpg"
out_path="testOutput.jpg"
image=cv2.imread(in_path,cv2.IMREAD_COLOR)
outList=[]
for i in range(3):
    print("size=%d ..."%case[i])
    outList.append(gauss(case[i],image))
cv2.imwrite(out_path,np.round(np.clip(outList[2],0,255)).astype(np.uint8))

#可视化

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"]=False

def toRGB(x):
    u8=np.round(np.clip(x,0,255)).astype(np.uint8)
    return cv2.cvtColor(u8,cv2.COLOR_BGR2RGB)

imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
resList=[]
for i in range(3):
    resList.append(toRGB(outList[i]))

y0=224
x0=352
S=416

fig,axes=plt.subplots(2,4,figsize=(14,7))
axes[0][0].imshow(imgRGB)
axes[0][0].set_title("original")
axes[0][0].axis("off")
axes[1][0].imshow(imgRGB[y0:y0+S,x0:x0+S])
axes[1][0].set_title("original crop")
axes[1][0].axis("off")
for i in range(3):
    axes[0][i+1].imshow(resList[i])
    axes[0][i+1].set_title("size="+str(case[i]))
    axes[0][i+1].axis("off")
    axes[1][i+1].imshow(resList[i][y0:y0+S,x0:x0+S])
    axes[1][i+1].set_title("size="+str(case[i])+" crop")
    axes[1][i+1].axis("off")
fig.tight_layout()
fig.savefig("testOutput_figure.png",dpi=130)
plt.show()

#输出结果
