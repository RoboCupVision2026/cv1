import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make(k, s):
    r = k // 2
    a = []
    for y in range(k):
        row = []
        for x in range(k):
            dx = x - r
            dy = y - r
            v = math.exp(-(dx**2 + dy**2) / (2 * s**2))
            row.append(v)
        a.append(row)
    t = sum(sum(row) for row in a)
    for y in range(k):
        for x in range(k):
            a[y][x] /= t
    return a


if __name__ == "__main__":
    for k, s in [(3, 1.0), (5, 1.0), (9, 2.0)]:
        a = make(k, s)
        fig, ax = plt.subplots()
        im = ax.imshow(a, cmap="viridis", vmin=0)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Manual: k={k}, sigma={s}")
        fig.tight_layout()
        fig.savefig(f"results/manual_k{k}_s{s}.png", dpi=160)
        plt.close(fig)
