from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
import SimpleITK as sitk
import numpy as np
from utils import save_nii_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_subdirectory(data_dir):
    subdirectories = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    return (
        max(subdirectories, key=lambda x: os.path.getmtime(x))
        if subdirectories
        else "No subdirectories found"
    )


def cal_metrics(pred, gt, save_path, method):
    """
    Calculate the metrics between the prediction and the ground truth.
    :param pred: The prediction
    :param gt: The ground truth
    :return: The metrics
    """
    save_path = os.path.join(save_path, method)
    psnr = PeakSignalNoiseRatio(data_range=gt.max() - gt.min()).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=gt.max() - gt.min()).to(device)
    psnr_val = psnr(pred, gt)
    ssim_val = ssim(pred, gt)
    psnr = psnr_val.item()
    ssim = ssim_val.item()
    error_map = torch.abs(pred - gt)

    save_nii_image(error_map, os.path.join(save_path, "error_map.nii"))
    save_nii_image(pred, os.path.join(save_path, "pred.nii"))
    save_nii_image(gt, os.path.join(save_path, "gt.nii"))

    # save psnr and ssim to txt file
    with open(os.path.join(save_path, "Metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr:.8f}\n")
        f.write(f"SSIM: {ssim:.8f}\n")

    return psnr, ssim


if __name__ == "__main__":
    mask = True
    clip_threshold = True
    clip_roi = False

    task = "LACT"
    method = "DPS-ALL"
    degree = 120
    data = "L004_flat_fan_projections"
    slice_index = 30
    dps_weight = 0.015

    # data_path = f"experiments/sample/{method}/{task}-{degree}-{method}-{data}-{slice_index}-{dps_weight}"
    # data_path = find_latest_subdirectory(data_path)

    # data_path = "./experiments/sample/DPER/DPER-120-30-0.02-2024-04-21-15-47-06-651288"
    data_path = "./experiments/sample/DPS/DPS-120-30-0.025"

    gt_path = os.path.join(data_path, "fbp_all.nii")
    fbp_lv_path = os.path.join(data_path, "fbp_la.nii")
    # recon_path = os.path.join(data_path, "NeRF/x_INR_DC_0.nii")
    recon_path = os.path.join(data_path, "recon.nii")
    recon_path = os.path.join(data_path, "x_0.nii")
    
    recon_path = "./experiments/sample/DPER/DPER-120-30-0.02-2024-04-21-15-47-06-651288/NeRF/x_INR_DC_950.nii"

    # dps_path = "experiments/sample/DPS/DPS-120-30-0.015/x_0.nii"

    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    fbp_lv = sitk.GetArrayFromImage(sitk.ReadImage(fbp_lv_path))
    recon = sitk.GetArrayFromImage(sitk.ReadImage(recon_path))

    if mask:
        mask_path = os.path.join("experiments/L004-30-0.7-mask.nii")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        mask = 1 - mask
        gt = gt * mask
        recon = recon * mask

    if clip_threshold:
        threshold = 0.05
        gt[gt < threshold] = threshold
        recon[recon < threshold] = threshold

    if clip_roi:
        x_left = 60
        x_right = 470
        # y_up = 110
        # y_down = 440
        y_up = 60
        y_down = 470
        gt = gt[y_up:y_down, x_left:x_right]
        fbp_lv = fbp_lv[y_up:y_down, x_left:x_right]
        recon = recon[y_up:y_down, x_left:x_right]

    gt = torch.tensor(gt).unsqueeze(0).unsqueeze(0).to(device)
    fbp_lv = torch.tensor(fbp_lv).unsqueeze(0).unsqueeze(0).to(device)
    recon = torch.tensor(recon).unsqueeze(0).unsqueeze(0).to(device)

    result = cal_metrics(recon, gt, data_path, method)
    print(f"{method}: ", result)
    result = cal_metrics(fbp_lv, gt, data_path, "FBP_LV")
    print("FBP_LV: ", result)
