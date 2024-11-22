import ct
from pathlib import Path
import utils
import numpy as np
import torch
import os

data_path = "/mnt/raid5/dch/code/DiffusionNerfProj/1_LACT/data/our_pure/helix2fan/out/L004/L004_flat_fan_projections.tif"
slice_index = 30
task = "LACT"
degree = 90

data_path = Path(data_path)
projections, meta_data = utils.load_tiff_stack_with_metadata(data_path)
projections = np.copy(np.flip(projections[:, :, slice_index], axis=1))
view_num, det_count = projections.shape


if task == "LACT":
    view_avai_num = int(view_num * (degree / 360))
    view_available = np.linspace(0, view_avai_num, view_avai_num, endpoint=False, dtype=int)

elif task == "SVCT":
    view_available = np.linspace(0, view_num, degree, endpoint=False, dtype=int)


voxel_size = 0.6
vox_scaling = 1 / voxel_size
scaling_factor = 20

projections = torch.from_numpy(projections).float().unsqueeze(0).unsqueeze(0).to("cuda")
projections = projections * vox_scaling * scaling_factor
measurement = projections[:, :, view_available, :].clone().detach()
print("Measurement input:", measurement.shape)


radon_torch_mod = ct.CT_Clinical_uni(
    meta_data=meta_data,
    img_width=int(meta_data["dsd"]),
    # img_width=512,
    view_limited_idx=view_available,
    det_count=det_count,
    voxel_size=voxel_size,
)
fbp_la = radon_torch_mod.FBP_LV(measurement, filter_name="hann")
fbp_all = radon_torch_mod.FBP_FV(projections, filter_name="hann")

mu_water = meta_data["hu_factor"] * scaling_factor
fbp_la = 1000 * ((fbp_la - mu_water) / mu_water)
fbp_all = 1000 * ((fbp_all - mu_water) / mu_water)


pad_start = (int(meta_data["dsd"]) - 512) // 2
pad_end = pad_start + 512

# fbp_la = fbp_la[:, :, pad_start:pad_end, pad_start:pad_end]
# fbp_all = fbp_all[:, :, pad_start:pad_end, pad_start:pad_end]


save_root = "./results/0424"
utils.save_nii_image(measurement.squeeze(), os.path.join(save_root, "measurement.nii"))
utils.save_nii_image(fbp_la, os.path.join(save_root, f"fbp_{task}.nii"))
utils.save_nii_image(fbp_all, os.path.join(save_root, "fbp_all.nii"))
