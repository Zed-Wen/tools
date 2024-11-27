import nibabel as nib
import imageio
import os
import numpy as np

# 本py用于将nii.gz转换为png保存

def get_file_names(directory):
    file_names = []
    # os.listdir()返回目录中的所有文件和目录名
    for entry in os.listdir(directory):
        # os.path.join用于路径拼接，检查每个entry是否为文件
        if os.path.isfile(os.path.join(directory, entry)):
            file_names.append(entry)
    return file_names

def nii_to_png(nii_path, png_save_path, png_file_name):
    # 确保保存图像的目录存在
    if not os.path.exists(png_save_path):
        os.makedirs(png_save_path)
    
    # 读取nii文件
    nii = nib.load(nii_path)
    nii_fdata = nii.get_fdata()

    # 顺时针旋转90度
    nii_fdata = np.rot90(nii_fdata, k=-1, axes=(1, 0))
    nii_fdata = np.flipud(nii_fdata)
    
    # 检查数据维度，确保它是二维的
    if len(nii_fdata.shape) != 2:
        raise ValueError("The nii file is not a 2D array.")

    # 正规化数据到0-255范围并转换为uint8
    min_val = nii_fdata.min()
    max_val = nii_fdata.max()
    nii_fdata_normalized = (nii_fdata - min_val) / (max_val - min_val) * 255
    nii_fdata_uint8 = nii_fdata_normalized.astype(np.uint8)

    # 保存二维数组为PNG图像
    imageio.imwrite(os.path.join(png_save_path, png_file_name), nii_fdata_uint8)


# 使用————————————————————————————————————————————————————————————————————————————————————————————————————————
nii_file_folder = 'niidata' # 输入文件夹
output_folder = 'pngdata'    # 输出文件夹

files = get_file_names(nii_file_folder)
for file in files:
    nii_file_path = os.path.join(nii_file_folder,file)  # 你的.nii文件路径
    filetype = file[:-7]
    nii_to_png(nii_file_path, output_folder, f'{filetype}.png')