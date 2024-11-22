import torch
import numpy as np
import torch_radon
from torchvision.transforms import Resize

class CT_Clinical:
    def __init__(self, meta_data, img_width, view_available, det_count, voxel_size, circle=False):
        voxel_size = voxel_size
        vox_scaling = 1 / voxel_size
        angles_all = np.array(meta_data["angles"])[: meta_data["rotview"]] + (np.pi / 2)
        angles_LA = angles_all[:view_available]
        self.img_width = img_width
        self.radon_all = torch_radon.RadonFanbeam(
            resolution=img_width,
            angles=angles_all,
            source_distance=vox_scaling * meta_data["dso"],
            det_distance=vox_scaling * meta_data["ddo"],
            det_count=det_count,
            det_spacing=vox_scaling * meta_data["du"],
            clip_to_circle=circle,
        )
        self.radon_LA = torch_radon.RadonFanbeam(
            resolution=img_width,
            angles=angles_LA,
            source_distance=vox_scaling * meta_data["dso"],
            det_distance=vox_scaling * meta_data["ddo"],
            det_count=det_count,
            det_spacing=vox_scaling * meta_data["du"],
            clip_to_circle=circle,
        )

    def A_LA(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_LA.forward(x)
        return sino

    def A_all(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_all.forward(x)
        return sino

    def FBP_LA(self, y, filter_name="ramp"):
        sino_filtered = self.radon_LA.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LA.backprojection(sino_filtered)
        return recon

    def FBP_all(self, y, filter_name="ramp"):
        sino_filtered = self.radon_all.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_all.backprojection(sino_filtered)
        return recon

    def BP_LA(self, y):
        recon = self.radon_LA.backprojection(y)
        return recon

    def BP_all(self, y):
        recon = self.radon_all.backprojection(y)
        return recon


class CT_Clinical_uni:
    """
    Uniform case for clinical CT with limited views (limited-angle or sparse-view) and full views.
    """

    def __init__(self, meta_data, img_width, view_limited_idx, det_count, voxel_size, circle=False):
        voxel_size = voxel_size
        vox_scaling = 1 / voxel_size

        angles_FV = np.array(meta_data["angles"])[: meta_data["rotview"]] + (np.pi / 2)
        angles_LV = angles_FV[view_limited_idx]

        self.img_width = img_width
        self.radon_FV = torch_radon.RadonFanbeam(
            resolution=img_width,
            angles=angles_FV,
            source_distance=vox_scaling * meta_data["dso"],
            det_distance=vox_scaling * meta_data["ddo"],
            det_count=det_count,
            det_spacing=vox_scaling * meta_data["du"],
            clip_to_circle=circle,
        )
        self.radon_LV = torch_radon.RadonFanbeam(
            resolution=img_width,
            angles=angles_LV,
            source_distance=vox_scaling * meta_data["dso"],
            det_distance=vox_scaling * meta_data["ddo"],
            det_count=det_count,
            det_spacing=vox_scaling * meta_data["du"],
            clip_to_circle=circle,
        )

    def A_LV(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_LV.forward(x)
        return sino

    def A_FV(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_FV.forward(x)
        return sino

    def FBP_LV(self, y, filter_name="ramp"):
        sino_filtered = self.radon_LV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LV.backprojection(sino_filtered)
        return recon

    def FBP_FV(self, y, filter_name="ramp"):
        sino_filtered = self.radon_FV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_FV.backprojection(sino_filtered)
        return recon

    def BP_LV(self, y):
        recon = self.radon_LV.backprojection(y)
        return recon

    def BP_FV(self, y):
        recon = self.radon_FV.backprojection(y)
        return recon


class CT_Clinical_uni_v2:
    """
    Uniform case for clinical CT with limited views (limited-angle or sparse-view) and full views.
    Using torch-radon-v2.
    """

    def __init__(self, img_width, view_limited_idx, circle=False):

        angles_FV = np.linspace(-np.pi / 2 * 1, np.pi / 2 * 3, 640, endpoint=False)
        # angles_FV = np.linspace(0, 2*np.pi, 640, endpoint=False)
        angles_LV = angles_FV[view_limited_idx]

        self.img_width = img_width
        self.radon_FV = torch_radon.FanBeam(
            det_count=img_width,
            angles=angles_FV,
            src_dist=1075,
            det_dist=1075,
        )
        self.radon_LV = torch_radon.FanBeam(
            det_count=img_width,
            angles=angles_LV,
            src_dist=1075,
            det_dist=1075,
        )

    def A_LV(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_LV.forward(x)
        return sino

    def A_FV(self, x):
        if x.shape[2] < 512:
            print("Resizing to 512")
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            # print("Padding to ", self.img_width)
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_FV.forward(x)
        return sino

    def FBP_LV(self, y, filter_name="ramp"):
        sino_filtered = self.radon_LV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LV.backward(sino_filtered)
        return recon

    def FBP_FV(self, y, filter_name="ramp"):
        sino_filtered = self.radon_FV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_FV.backward(sino_filtered)
        return recon

    def BP_LV(self, y):
        recon = self.radon_LV.backward(y)
        return recon

    def BP_FV(self, y):
        recon = self.radon_FV.backward(y)
        return recon


class CT_Clinical_carterbox:
    def __init__(self, meta_data, img_width, view_available, det_count, voxel_size):
        self.img_width = img_width
        vox_scaling = 1 / voxel_size
        angles_all = np.array(meta_data["angles"])[: meta_data["rotview"]] + (np.pi / 2)
        angles_LA = angles_all[:view_available]
        volume = torch_radon.Volume2D(center=(0, 0), voxel_size=(voxel_size, voxel_size))
        volume.set_size(self.img_width, self.img_width)
        self.radon_all = torch_radon.FanBeam(
            det_count=det_count,
            angles=angles_all,
            src_dist=vox_scaling * meta_data["dso"],
            det_dist=vox_scaling * meta_data["ddo"],
            det_spacing=vox_scaling * meta_data["du"],
            volume=volume,
        )
        self.radon_LA = torch_radon.FanBeam(
            det_count=det_count,
            angles=angles_LA,
            src_dist=vox_scaling * meta_data["dso"],
            det_dist=vox_scaling * meta_data["ddo"],
            det_spacing=vox_scaling * meta_data["du"],
            volume=volume,
        )

    def A_LA(self, x):
        if x.shape[2] != 512:
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        if x.shape[2] != self.img_width:
            x_new = torch.zeros(x.shape[0], x.shape[1], self.img_width, self.img_width).to(x.device)
            pad_start = (self.img_width - x.shape[2]) // 2
            pad_end = pad_start + x.shape[2]
            x_new[:, :, pad_start:pad_end, pad_start:pad_end] = x
            x = x_new
        sino = self.radon_LA.forward(x)
        return sino

    def A_all(self, x):
        sino = self.radon_all.forward(x)
        return sino

    def FBP_LA(self, y, filter_name="ramp"):
        sino_filtered = self.radon_LA.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LA.backward(sino_filtered)
        return recon

    def FBP_all(self, y, filter_name="ramp"):
        sino_filtered = self.radon_all.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_all.backward(sino_filtered)
        return recon

    def BP_LA(self, y):
        recon = self.radon_LA.backward(y)
        return recon

    def BP_all(self, y):
        recon = self.radon_all.backward(y)
        return recon
