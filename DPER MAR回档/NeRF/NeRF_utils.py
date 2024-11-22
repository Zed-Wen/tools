import numpy as np
import torch
import math
import SimpleITK as sitk
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# from cil.framework import AcquisitionGeometry,ImageGeometry,ImageData
# from cil.plugins.tigre import FBP
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def fourier_feature_mapping(xyz, B):
    """
    fourier feature mapping function
    """
    xyz_embedding_cos = torch.cos(xyz.float() @ B.float() * 2 * math.pi)
    xyz_embedding_sin = torch.sin(xyz.float() @ B.float() * 2 * math.pi)
    xyz_embedding = torch.cat([xyz_embedding_cos, xyz_embedding_sin], dim=1)
    return xyz_embedding


def position_encoding_mapping(xyz, lbase, levels):
    """
    position encoding mapping function
    """
    pe_list = []
    for i in range(levels):
        temp_value = xyz.float() * lbase**i * math.pi
        pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
    return torch.stack(pe_list, 1)


def normalization(data):
    v_max = np.max(data)
    v_min = np.min(data)
    return (data - v_min) / (v_max - v_min)


def build_coordinate_train(L, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    x = np.linspace(-0.5, 0.5, L)
    y = np.linspace(-0.5, 0.5, L)
    x, y = np.meshgrid(x, y, indexing="ij")  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)
    xy = xy @ trans_matrix.T  # (L*L, 2) @ (2, 2)
    xy = xy.reshape(L, L, 2)
    return xy


def build_coordinate_train_fan(xy, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    L, D_2, _ = xy.shape
    xy = xy.reshape(-1, 2)  # (L*2D, 2)
    xy = xy @ trans_matrix.T  # (L*2D, 2) @ (2, 2) -> (L*2D, 2)
    xy = xy.reshape(L, D_2, 2)  # (L, 2D, 2)
    return xy


def fan_coordinate(fan_angle, D):
    origin_x = 1
    origin_y = 0
    x = np.linspace(-1, 1, int(2 * D)).reshape(-1, 1)  # (2D, ) -> (2D, 1)
    y = np.zeros_like(x)  # (2D, 1)
    xy_temp = np.concatenate((x, y), axis=-1)  # (2D, 2)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)  # (2D, 3)

    L = len(fan_angle)
    xy = np.zeros(shape=(L, int(2 * D), 2))  # (L, 2D, 2)
    for i in range(L):
        fan_angle_rad = np.deg2rad(fan_angle[L - i - 1])
        M = np.array(
            [
                [
                    np.cos(fan_angle_rad),
                    -np.sin(fan_angle_rad),
                    -1 * origin_x * np.cos(fan_angle_rad) + origin_y * np.sin(fan_angle_rad) + origin_x,
                ],
                [
                    np.sin(fan_angle_rad),
                    np.cos(fan_angle_rad),
                    -1 * origin_x * np.sin(fan_angle_rad) - origin_y * np.cos(fan_angle_rad) + origin_y,
                ],
                [0, 0, 1],
            ]
        )
        temp = xy_temp @ M.T  # (2D, 3) @ (3, 3) -> (2D, 3)
        xy[i, :, :] = temp[:, :2]
    return xy


def psnr(image, ground_truth):
    N = image.shape[0]
    res = []
    for k in range(N):
        mse = np.mean((image[k] - ground_truth[k]) ** 2)
        if mse == 0.0:
            return float("inf")
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(20 * np.log10(data_range) - 10 * np.log10(mse))
    return np.array(res)


def ssim(image, ground_truth):
    N = image.shape[0]
    res = []
    for k in range(N):
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(structural_similarity(image[k], ground_truth[k], data_range=data_range))
    res = np.array(res)
    return res


def iradon_transform(sinograms, num_angles):
    image_lst = []
    batch_size = sinograms.shape[0]
    theta = np.linspace(0, 180, num_angles, endpoint=False)
    for index in range(batch_size):
        recon_fbp = iradon(sinograms[index, 0], theta, circle=False, filter_name="ramp")
        image_lst.append(recon_fbp)
    image_array = np.array(image_lst)[:, None, :, :]
    images = torch.from_numpy(image_array)
    return images


class TVLoss_intensity(nn.Module):
    def __init__(self):
        super(TVLoss_intensity, self).__init__()

    def forward(self, x):
        row, col = x.shape[0], x.shape[1]
        dx = torch.nan_to_num(torch.abs(x[1:, :] - x[: row - 1, :]))
        dy = torch.nan_to_num(torch.abs(x[:, 1:] - x[:, : col - 1]))
        tv_loss = torch.sum(dx) + torch.sum(dy)
        # tv_loss = (torch.mean(dx)+torch.mean(dy) +
        #            torch.mean(dz) / (3*torch.max(x)))
        return tv_loss / (row * col)


class TVLoss_proj(nn.Module):
    def __init__(self):
        super(TVLoss_proj, self).__init__()

    def forward(self, projection):
        N, sample_N = projection.shape[0], projection.shape[1]

        # dx = torch.nan_to_num(torch.abs(projection[1:, :] - projection[: row - 1, :]))
        dy = torch.nan_to_num(torch.abs(projection[:, 1:] - projection[:, : sample_N - 1]))
        # tv_loss = torch.sum(dx) + torch.sum(dy)
        # print(torch.sum(dx), torch.sum(dy))
        tv_loss = torch.sum(dy) / (N * sample_N)

        return tv_loss


if __name__ == "__main__":
    sinogram_path = "data/sino_95_0.nii"
    output_path = "output/Recon_95_0_FBP.nii"
    recon_matlab_path = "output/Recon_90_matlab.nii"
    recon_nerf_path = "output/recon_2000.nii"
    # recon_nerf_path = r'D:\Study\LACT\DDPM_CT\output\CT_1\NO_recon_0_newradon.nii'
    gt_path = "data/gt_95.nii"
    # gt_path = r'D:\Study\LACT\DDPM_CT\data\CT_train_1_resize_128.nii'
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    sino = sitk.GetArrayFromImage(sitk.ReadImage(sinogram_path))
    recon_matlab = sitk.GetArrayFromImage(sitk.ReadImage(recon_matlab_path)).T
    recon_nerf = sitk.GetArrayFromImage(sitk.ReadImage(recon_nerf_path))
    # gpu = 0
    # DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    # print(DEVICE)

    # skimage的方法
    sino_input = np.expand_dims(sino, axis=(0, 1))
    n_angles = sino.shape[-1]
    recon_img_skimage = iradon_transform(sino_input, n_angles)
    psnr_fbp_gt_skimage = psnr(recon_img_skimage[0].numpy(), gt[None])
    ssim_fbp_gt_skimage = ssim(recon_img_skimage[0].numpy(), gt[None])
    print("skimage recon method:{},{}".format(psnr_fbp_gt_skimage, ssim_fbp_gt_skimage))
    recon_skimage = sitk.GetImageFromArray(recon_img_skimage[0, 0, :, :])
    # sitk.WriteImage(recon_skimage, 'output/recon_skimage.nii')
    # plt.imshow(recon_img_skimage[0,0,:,:])
    # plt.show()

    # matlab方法
    # print(recon_matlab[None].shape)
    # psnr_fbp_gt_matlab = psnr(recon_matlab[None],gt[None])
    # print('matlab recon method:{}'.format(psnr_fbp_gt_matlab))
    # plt.imshow(recon_matlab[:,:])
    # plt.show()
    # CIL 的方法

    # train nerf 方法
    print(recon_nerf[None].shape)
    psnr_fbp_gt_nerf = psnr(recon_nerf[None], gt[None])
    ssim_fbp_gt_nerf = ssim(recon_nerf[None], gt[None])
    print("nerf recon method:{},{}".format(psnr_fbp_gt_nerf, ssim_fbp_gt_nerf))
    plt.imshow(recon_nerf[:, :])
    # plt.imshow(recon_nerf[0])
    plt.show()
