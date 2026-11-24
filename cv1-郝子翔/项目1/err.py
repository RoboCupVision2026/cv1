# 两个高斯核对应位置做差，画出绝对误差
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gaosihe import make as manual
from cv import make as opencv

for k, s in [(3, 1.0), (5, 1.0), (9, 2.0)]:
    m = np.array(manual(k, s))
    c = opencv(k, s)
    d = np.abs(m - c)
    fig, ax = plt.subplots()
    im = ax.imshow(d, cmap="magma", vmin=0)
    fig.colorbar(im, ax=ax, label="Absolute error")
    ax.set_title(f"k={k}, sigma={s}\nMax error: {d.max():.3e}")
    fig.tight_layout()
    fig.savefig(f"results/err_k{k}_s{s}.png", dpi=160)
    plt.close(fig)
