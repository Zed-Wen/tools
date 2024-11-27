import torch as th
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import matplotlib.pyplot as plt


def save_nii_image(image, save_path):
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, th.Tensor):
        if image.requires_grad:
            image = image.detach()
        image = image.float().squeeze().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.squeeze()

    sitk_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_image, save_path)


def normalize(image):
    # 确定原始值域的最小值和最大值
    min_value = image.min()
    max_value = image.max()

    # 归一化处理
    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image

def cal_metrics(pred, gt, save_path):
    """
    Calculate the metrics between the prediction and the ground truth.
    :param pred: The prediction
    :param gt: The ground truth
    :return: The metrics
    """
    if save_path.endswith(".nii"):
        save_path = save_path[:-4]
    os.makedirs(save_path, exist_ok=True)

    device = gt.device
    psnr = PeakSignalNoiseRatio(data_range=gt.max() - gt.min()).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=gt.max() - gt.min()).to(device)
    psnr_val = psnr(pred, gt)
    ssim_val = ssim(pred, gt)
    psnr = psnr_val.item()
    ssim = ssim_val.item()
    error_map = pred - gt
    error_map_abs = th.abs(pred - gt)

    save_nii_image(error_map, os.path.join(save_path, "error_map.nii"))
    save_nii_image(error_map_abs, os.path.join(save_path, "error_map_abs.nii"))
    save_nii_image(pred, os.path.join(save_path, "pred.nii"))
    save_nii_image(gt, os.path.join(save_path, "gt.nii"))

    # save psnr and ssim to txt file
    with open(os.path.join(save_path, "Metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr:.8f}\n")
        f.write(f"SSIM: {ssim:.8f}\n")

    return psnr, ssim


def cal_metrics_considering_mask(pred, gt, mask, save_path):
    """
    Calculate the metrics between the prediction and the ground truth.
    :param pred: The prediction
    :param gt: The ground truth
    :return: The metrics
    """
    # plt.imshow(mask.squeeze(0).squeeze(0).cpu(), cmap = 'gray')
    # plt.show()
    device = gt.device
    pred[mask == 1] = pred.max()
    gt[mask == 1] = gt.max()
    plt.imshow(pred.squeeze(0).squeeze(0).cpu(), cmap = 'gray')
    plt.show()
    psnr = PeakSignalNoiseRatio(data_range=gt.max() - gt.min()).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=gt.max() - gt.min()).to(device)
    psnr_val = psnr(pred, gt)
    ssim_val = ssim(pred, gt)
    psnr = psnr_val.item()
    ssim = ssim_val.item()
    error_map = pred - gt
    error_map_abs = th.abs(pred - gt)

    save_nii_image(error_map, os.path.join(save_path, "error_map.nii"))
    save_nii_image(error_map_abs, os.path.join(save_path, "error_map_abs.nii"))
    save_nii_image(pred, os.path.join(save_path, "pred.nii"))
    save_nii_image(gt, os.path.join(save_path, "gt.nii"))

    # save psnr and ssim to txt file
    with open(os.path.join(save_path, "Metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr:.8f}\n")
        f.write(f"SSIM: {ssim:.8f}\n")


    return psnr, ssim



# example use
gt_path = "gt.nii"
pre_path = 'dper.nii'
mask_path = 'mask.nii'
# pre = normalize(th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(pre_path))).unsqueeze(0).unsqueeze(0))
# gt = normalize(th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(gt_path))).unsqueeze(0).unsqueeze(0))
# pre = th.flip(th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(pre_path))).unsqueeze(0).unsqueeze(0),[3])
pre = th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(pre_path))).unsqueeze(0).unsqueeze(0)
# pre = (pre + 1) /2
gt = th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(gt_path))).unsqueeze(0).unsqueeze(0)
# gt = th.flip(gt,[3])
mask = th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(mask_path))).unsqueeze(0).unsqueeze(0)

# psnr,ssim = cal_metrics(pre, gt, "metrics")
psnr,ssim = cal_metrics_considering_mask(pre, gt, mask, "metrics")

print("PSNR,SSIM respectively:{}, {}".format(psnr,ssim))