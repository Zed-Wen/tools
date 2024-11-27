import numpy as np
import scipy.io as sio
import SimpleITK as sitk

# 将mat变成npy使用以下代码
# 读取 .mat 文件
mat_data = sio.loadmat('SampleMasks.mat')

# 获取数据（假设数组名为 'proj'）
img_array = mat_data['CT_samples_bwMetal']

# 转为.nii文件
sitk.WriteImage(sitk.GetImageFromArray(img_array), "save.nii")

# 转换为 .npy 文件
np.save('recon256.npy', img_array)


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# # 将mat变成npy使用以下代码
# img256 = np.load('img256.npy')
# sio.savemat("img256.mat", {'img': img256})






