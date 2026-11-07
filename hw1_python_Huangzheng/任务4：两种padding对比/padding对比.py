import numpy as np
import cv2
import math

size=9
#核的尺寸

def getKe2D(size):
    sg=0.3*((size-1)*0.5-1)+0.8
    sig2=2*sg*sg
    center=size//2
    K=np.zeros((size,size),dtype=np.float64)
    Sum=0.0
    for i in range(size):
        for j in range(size):
            dy=i-center
            dx=j-center
            K[i][j]=math.exp(-(dx*dx+dy*dy)/sig2)
            Sum+=K[i][j]
    for i in range(size):
        for j in range(size):
            K[i][j]=K[i][j]/Sum
    return K

def padImage(img,p,mode):
    height,width=img.shape[:2]
    out=np.zeros((height+2*p,width+2*p,3),dtype=np.float64)
    for i in range(height+2*p):
        for j in range(width+2*p):
            si=i-p
            sj=j-p
            if(mode=="zero"):
                if(si<0 or si>height-1 or sj<0 or sj>width-1):
                    continue
    #边缘补零
            else:
                if(si<0): si=0
                if(si>height-1): si=height-1
                if(sj<0): sj=0
                if(sj>width-1): sj=width-1
            out[i][j]=img[si][sj]
    #边缘复制
    return out

def convPad(img,K,mode):
    size=K.shape[0]
    center=size//2
    height,width=img.shape[:2]
    padded=padImage(img,center,mode)
    out=np.zeros((height,width,3),dtype=np.float64)
    for i in range(height):
        for j in range(width):
            cnt=[0,0,0]
            for ky in range(size):
                for kx in range(size):
                    wt=K[ky][kx]
                    for c in range(3):
                        cnt[c]+=padded[i+ky][j+kx][c]*wt
            for c in range(3):
                out[i][j][c]=cnt[c]
    return out

#读图片

in_path=r"testInput.jpg"
image=cv2.imread(in_path,cv2.IMREAD_COLOR)
s=400.0/max(image.shape[0],image.shape[1])
image=cv2.resize(image,(int(image.shape[1]*s),int(image.shape[0]*s)))
print("工作尺寸:",image.shape)

#两种padding分别滤波

K=getKe2D(size)
rZero=convPad(image,K,"zero")
rEdge=convPad(image,K,"edge")
pZero=padImage(image,30,"zero")
pEdge=padImage(image,30,"edge")

#量化对比

b=5
def edgeMean(x):
    return (np.mean(x[:b,:,:])+np.mean(x[-b:,:,:])+np.mean(x[:,:b,:])+np.mean(x[:,-b:,:]))/4
print("边缘%d像素平均亮度: 原图=%.1f 补0=%.1f 复制边缘=%.1f"%(b,edgeMean(image.astype(np.float64)),edgeMean(rZero),edgeMean(rEdge)))
print("两种结果最大差=%.1f 灰阶"%np.abs(rZero-rEdge).max())

#拼图

def toU8(x):
    return np.round(np.clip(x,0,255)).astype(np.uint8)

def fit(x,n=300):
    return cv2.resize(toU8(x),(n,n),interpolation=cv2.INTER_NEAREST)

def markPad(padded,p):
    vis=padded.copy()
    for k in range(3):
        vis[p+k,:,:]=[0,0,255]
        vis[padded.shape[0]-p-1-k,:,:]=[0,0,255]
        vis[:,p+k,:]=[0,0,255]
        vis[:,padded.shape[1]-p-1-k,:]=[0,0,255]
    return vis

d=np.abs(rZero-rEdge)*4
panels=[("original",image),
        ("pad zero (p=30)",markPad(pZero,30)),
        ("pad edge (p=30)",markPad(pEdge,30)),
        ("blur + zero",rZero),
        ("blur + edge",rEdge),
        ("abs(zero-edge)*4 corner x5",d[0:60,0:60])]

N=300
GAP=10
LAB=34
canvas=np.full((2*(N+LAB)+3*GAP,3*N+4*GAP,3),255,dtype=np.uint8)
for idx,(title,arr) in enumerate(panels):
    r=idx//3
    c=idx%3
    y=GAP+r*(N+LAB+GAP)
    x=GAP+c*(N+GAP)
    cv2.putText(canvas,title,(x+2,y+24),cv2.FONT_HERSHEY_SIMPLEX,0.62,(20,20,20),1,cv2.LINE_AA)
    canvas[y+LAB:y+LAB+N,x:x+N]=fit(arr)

ok=cv2.imwrite("padding_result.png",canvas)
print("imwrite 返回:",ok)
print("已保存 padding_result.png",canvas.shape)
cv2.imshow("padding",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


