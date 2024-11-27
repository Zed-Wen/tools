import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat

# 本py用于将.mat文件读取并转换为numpy进行plot可视化（代码普适性较差，待改）

# 改这些
folder = '000001_01_01'
data_index = 103
metal_mask_index = 5

# 读取mat
raw_data_path = 'C:/Users/92887/Desktop/adn-master/adn-master/data/deep_lesion/raw'
train_data_path = 'C:/Users/92887/Desktop/adn-master/adn-master/data/deep_lesion/train'  # 替换为你的图片路径
Xma_data_name = f'{data_index}/{metal_mask_index}/Xma.mat'
XLI_data_name = f'{data_index}/{metal_mask_index}/XLI.mat'
Sma_data_name = f'{data_index}/{metal_mask_index}/Sma.mat'
SLI_data_name = f'{data_index}/{metal_mask_index}/SLI.mat'
Xgt_data_name = f'{data_index}/gt.mat'
# train_data_name = f'{data_index}/gt.mat'


# Xgt
mat_file_path = os.path.join(train_data_path, folder, Xgt_data_name)  # 替换为你的.mat文件路径
mat_data = loadmat(mat_file_path)
Xgt = mat_data['image']

# Xma
mat_file_path = os.path.join(train_data_path, folder, Xma_data_name)  # 替换为你的.mat文件路径
mat_data = loadmat(mat_file_path)
Xma = mat_data['image']

# XLI
mat_file_path = os.path.join(train_data_path, folder, XLI_data_name)  # 替换为你的.mat文件路径
mat_data = loadmat(mat_file_path)
XLI = mat_data['image']

# SLI
mat_file_path = os.path.join(train_data_path, folder, SLI_data_name)  # 替换为你的.mat文件路径
mat_data = loadmat(mat_file_path)
SLI = mat_data['image']

# Sma
mat_file_path = os.path.join(train_data_path, folder, Sma_data_name)  # 替换为你的.mat文件路径
mat_data = loadmat(mat_file_path)
Sma = mat_data['image']

plt.subplot(1,3,1)
plt.imshow(Xgt, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(Xma, cmap='gray', vmin = 0, vmax = 0.3)
plt.subplot(1,3,3)
plt.imshow(XLI, cmap='gray', vmin = 0, vmax = 0.3)
plt.show()

plt.subplot(1,2,1)
plt.imshow(SLI, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(Sma, cmap='gray')
plt.show()