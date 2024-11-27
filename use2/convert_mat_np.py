import numpy as np
import scipy.io as sio
import SimpleITK as sitk

# 将mat变成npy使用以下代码
# 读取 .mat 文件
mat_data = sio.loadmat('imgWater_local.mat')

print(mat_data['imgWater_local'].shape)

# 获取数据（假设数组名为 'proj'）
img_array = mat_data['imgWater_local']

img_array = img_array.transpose()

print(img_array.shape)

# 转为.nii文件
sitk.WriteImage(sitk.GetImageFromArray(img_array), "imgWater_local.nii")

# # # 转换为 .npy 文件
# # np.save('recon256.npy', img_array)


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# # nii 转 mat
# img = sitk.GetArrayFromImage(sitk.ReadImage("gt.nii"))
# print(img[199:205,199:205])

# # sio.savemat("gt.mat",{'gt':img})

