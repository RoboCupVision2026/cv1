import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make(k, s):
    g = cv2.getGaussianKernel(k, s, cv2.CV_64F)
    return g @ g.T


if __name__ == "__main__":
    for k, s in [(3, 1.0), (5, 1.0), (9, 2.0)]:
        a = make(k, s)
        fig, ax = plt.subplots()
        im = ax.imshow(a, cmap="viridis", vmin=0)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"OpenCV: k={k}, sigma={s}")
        fig.tight_layout()
        fig.savefig(f"results/opencv_k{k}_s{s}.png", dpi=160)
        plt.close(fig)
