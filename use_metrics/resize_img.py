import torch as th
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
from torchvision.transforms import Resize

def resize(image, target_size):
    """
    image: 必须是tensor,待resize原图
    target_size: 想resize的目标尺寸
    返回resize后的图片,以二维tensor格式
    """
    print("original size:",image.shape)
    while len(image.shape) < 4:   # # torchvision.transforms的Resize要求必须输入图片是（N,C,d1,d2,...,dk）shape的tensor，N和C不可少
        image = image.unsqueeze(0)
    while len(image.shape) > 4:
        image = image.squeeze()
    resize_f = Resize(target_size)
    resized_image = resize_f(image)
    resized_image = resized_image.squeeze().squeeze()
    print("processed size:",resized_image.shape)
    return resized_image

# example use
img_path = "gt.nii"
img = th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
resized_img = resize(img, 512)
sitk.WriteImage(sitk.GetImageFromArray(resized_img), "resized_img.nii")