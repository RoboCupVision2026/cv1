import math


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
