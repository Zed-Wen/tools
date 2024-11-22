import torch as th
import SimpleITK as sitk
import tifffile
import numpy as np
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os


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


def load_tiff_stack_with_metadata(file):
    """

    :param file: Path object describing the location of the file
    :return: a numpy array of the volume, a dict with the metadata
    """
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", '"')
    try:
        import json as js_ori

        metadata = js_ori.loads(metadata)
    except:
        print("The tiff file you try to open does not seem to have metadata attached.")
        metadata = None
    return data, metadata


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
