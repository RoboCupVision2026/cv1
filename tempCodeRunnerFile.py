for i, (size, sigma) in enumerate(params):
#     k1 = gauss_kernel(size, sigma)
#     k = cv2.getGaussianKernel(size, sigma)
#     k2 = k @ k.T
#     diff = np.abs(k1 - k2)
#     print(f"size = {size}, sigma = {sigma}")
#     print("Manual kernel:\n", k1)
#     print("OpenCV kernel:\n", k2)
#     print("Max error:", diff.max())
#     print()

#     vmax = max(k1.max(), k2.max())
#     a = axes[i, 0].imshow(
#         k1, cmap = "hot", vmin = 0, vmax = vmax
#     )
#     axes[i, 0].set_title(
#         f"Manual: {size}x{size}, sigma={sigma}"
#     )

#     b = axes[i, 1].imshow(
#         k2, cmap = "hot", vmin = 0, vmax = vmax
#     )
#     axes[i, 1].set_title("OpenCV")

#     c = axes[i, 2].imshow(diff, cmap = "viridis")
#     axes[i, 2].set_title("Absolute error")

#     fig.colorbar(a, ax = axes[i, 0])
#     fig.colorbar(b, ax = axes[i, 1])
#     fig.colorbar(c, ax = axes[i, 2])

# plt.tight_layout()
# plt.savefig("experiment2.png", dpi = 200)
# plt.show()



