import numpy as np
# from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# def interp2(X, Y, Z, Xq, Yq):
#     """
#     Interpolate values from a 2D grid (X, Y, Z) to specified query points (Xq, Yq).

#     Parameters:
#         X (ndarray): 2D array of X coordinates.
#         Y (ndarray): 2D array of Y coordinates.
#         Z (ndarray): 2D array of values corresponding to X and Y.
#         Xq (ndarray): 2D array of query points for X.
#         Yq (ndarray): 2D array of query points for Y.

#     Returns:
#         ndarray: Interpolated values at query points (Xq, Yq).
#     """
#     # Create a RectBivariateSpline for interpolation
#     interp_func = RectBivariateSpline(X[:, 0], Y[0, :], Z)

#     # Evaluate the function at query points
#     return interp_func(Xq, Yq)

# # Example usage
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Y = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
# Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Xq = np.array([[1.5, 2.5], [4.5, 5.5]])
# Yq = np.array([[15, 25], [45, 55]])

# result = interp2(X, Y, Z, Xq, Yq)
# result


img_all = np.load("proj256.npy")

# Plot original image
for i in range(img_all.shape[2]):
    plt.figure(1)
    plt.ion()  #打开交互模式
    plt.imshow(img_all[:,:,i], cmap="gray")
    plt.title(f"view: {i}")
    plt.colorbar()
    plt.show(block = False)
    plt.pause(0.1)
    plt.savefig(f"./proj/view_{i}")
    plt.clf()  #清除图像