import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import SimpleITK as sitk
import commentjson as json
import numpy as np
import tinycudann as tcnn
import torch
import pathlib
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.transform import iradon, radon
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import NeRF.NeRF_dataset as dataset


def train_fit(config_path, prior_image, save_path, device):
    with open(config_path) as config_file:
        config = json.load(config_file)

    h, L = config["file"]["h"], config["file"]["L"]
    train_params = config["train_fit"]

    out_path = os.path.join(save_path, "NeRF")
    model_save_path = os.path.join(save_path, "NeRF_checkpoint")

    for path in [out_path, model_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    pad_start = (L - h) // 2
    pad_end = pad_start + h
    prior_image_padding = torch.zeros((L, L)).to(device)
    prior_image_padding[pad_start:pad_end, pad_start:pad_end] = prior_image.to(device)

    train_loader = data.DataLoader(
        dataset=dataset.TrainData_fit(img=prior_image_padding),
        batch_size=train_params["batch_size"],
        shuffle=True,
    )
    test_loader = data.DataLoader(
        dataset=dataset.TestData_fit(img=prior_image_padding),
        batch_size=train_params["batch_size"],
        shuffle=False,
    )

    l2_loss = torch.nn.MSELoss()
    ms_ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1).to(device)

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=train_params["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=train_params["lr_decay_epoch"], gamma=train_params["lr_decay_coefficient"]
    )

    for e in tqdm(range(train_params["epoch"])):
        net.train()
        loss_train = 0
        for i, (xy, img) in enumerate(train_loader):
            xy = xy.to(device).float().view(-1, 2)
            img = img.to(device).float().view(-1, 1)
            img_pre = net(xy)
            loss = l2_loss(img_pre, img.to(img_pre.dtype))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step()
        avg_loss = loss_train / len(train_loader)
        print(
            f"(1. Prior Embedding) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.6f}"
        )

        if avg_loss < 0.00000000001:
            torch.save(net.state_dict(), os.path.join(model_save_path, "Prior_Embedding.pkl"))
            break
        if (e + 1) % train_params["summary_epoch"] == 0 or e == 0:
            with torch.no_grad():
                for i, (xy) in enumerate(test_loader):
                    img_pre = net(xy.to(device).float().view(-1, 2)).view(L, L)[
                        pad_start:pad_end, pad_start:pad_end
                    ]
                    img_pre_numpy = img_pre.cpu().float().numpy()
                    psnr_value = peak_signal_noise_ratio(
                        prior_image.view(h, h).cpu().numpy(),
                        img_pre_numpy,
                        data_range=prior_image.cpu().numpy().max() - prior_image.cpu().numpy().min(),
                    )
                    print("PSNR:", psnr_value)
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img_pre_numpy),
                        os.path.join(out_path, f"Prior_Embedding_epoch_{e + 1}.nii"),
                    )
                    torch.save(net.state_dict(), os.path.join(model_save_path, "Prior_Embedding.pkl"))
    img_pre = img_pre.to(device).view(1, 1, h, h)
    return img_pre


def train_refine(config_path, view_N, lv_sino, degrees_all, save_path, device, init=False):
    with open(config_path) as config_file:
        config = json.load(config_file)

    num_dv, L, h = config["file"]["num_dv"], config["file"]["L"], config["file"]["h"]
    train_params = config["train_refine"]

    pad_start = (L - h) // 2
    pad_end = pad_start + h

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")
    model_path = os.path.join(model_save_path, "Prior_Embedding.pkl")

    for path in [out_path, model_save_path]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    train_loader = data.DataLoader(
        dataset=dataset.TrainData_refine(
            view_N=view_N, sinogram=lv_sino, sample_N=train_params["sample_N"], degrees_all=degrees_all
        ),
        batch_size=train_params["batch_size"],
        shuffle=True,
    )
    test_loader = data.DataLoader(dataset=dataset.TestDataDirect(L=L), batch_size=1, shuffle=False)

    smooth_l1_loss = torch.nn.SmoothL1Loss()

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)

    if not init:
        net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=train_params["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for e in tqdm(range(train_params["epoch"])):
        net.train()
        loss_train = 0
        for i, (ray, projection_l) in enumerate(train_loader):
            # ray: 采集的射线的坐标，projection_l: 采集的射线的投影值总和，pre_intensity: 采集的射线的预测值
            ray = ray.to(device).float().view(-1, 2)  # (N, sample_N, L, 2)
            projection_l = projection_l.to(device).float()  # (N, sample_N)
            pre_intensity = net(ray).view(-1, train_params["sample_N"], L, 1)  # (N, sample_N, L, 1)
            projection_l_pre = torch.sum(pre_intensity, dim=2)  # (N, sample_N, 1, 1)
            projection_l_pre = projection_l_pre.squeeze(-1).squeeze(-1)  # (N, sample_N)

            loss = smooth_l1_loss(projection_l_pre, projection_l.to(projection_l_pre.dtype))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()
        avg_loss = loss_train / len(train_loader)

        if (e + 1) % 10 == 0:
            print(
                f"(2. INR Refine) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.6f}"
            )

        if (e + 1) % train_params["summary_epoch"] == 0:
            with torch.no_grad():
                for i, xy in enumerate(test_loader):
                    xy = xy.to(device).float().view(-1, 2)  # (h*w, 2)
                    img_pre = net(xy).view(L, L).float()
                    img_pre = img_pre[pad_start:pad_end, pad_start:pad_end].reshape(1, 1, h, h)
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img_pre.squeeze(0).squeeze(0).cpu().detach().numpy()),
                        os.path.join(out_path, f"INR_Refine_epoch_{e + 1}.nii"),
                    )

    return img_pre


def reprojection(config_path, view_N, degrees_all, lv_sino, save_path, device):
    with open(config_path) as config_file:
        config = json.load(config_file)

    num_dv, L = config["file"]["num_dv"], config["file"]["L"]
    batch_size = config["train_refine"]["batch_size"]

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")
    model_path = os.path.join(model_save_path, "INR_Refine.pkl")

    for path in [out_path, model_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    test_loader = data.DataLoader(
        dataset=dataset.TestData_refine(theta=num_dv, L=L),
        batch_size=batch_size,
        shuffle=False,
    )

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"],
    ).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    sin_pre = np.zeros(shape=(num_dv, L))
    with torch.no_grad():
        for i, ray in enumerate(test_loader):
            ray = ray.to(device).float().view(-1, 2)
            pre_intensity = net(ray).view(-1, L, L, 1)
            projection_l_pre = torch.sum(pre_intensity, dim=2).squeeze(-1).squeeze(-1)

            temp = projection_l_pre.cpu().detach().numpy()
            if i == 0:
                sin_pre = temp
            else:
                sin_pre = np.concatenate((sin_pre, temp), axis=0)

    theta_available = np.linspace(0, degrees_all, view_N, endpoint=False)
    theta_all = np.linspace(0, 180, num_dv, endpoint=False)
    locations = np.searchsorted(theta_all, theta_available)
    # locations = np.array(locations, dtype=int)
    sin_pre[locations] = lv_sino
    img_reprojection = iradon(sin_pre, theta=theta_all, circle=False)

    return sin_pre, img_reprojection


# =============== 联合优化 ===============
# TODO:
#  1. 需要重写
def train_joint_optim(config_path, prior_image, view_N, lv_sino, degrees_all, save_path, device):
    # Load config
    with open(config_path) as config_file:
        config = json.load(config_file)

    h, L, epoch, sample_N, batch_size = (
        config["file"]["h"],
        config["file"]["L"],
        config["train_refine"]["epoch"],
        config["train_refine"]["sample_N"],
        config["train_refine"]["batch_size"],
    )

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")

    for path in [out_path, model_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    model_path = os.path.join(model_save_path, "joint_optim.pkl")
    prior_image_clone = prior_image.clone()
    prior_image = torch.flip(prior_image, dims=[0])
    prior_image_padding = torch.zeros((2 * L, 2 * L)).to(device)
    prior_image_padding[235:-235, 235:-235] = prior_image.to(device)

    pad_start = (L - h) // 2
    pad_end = pad_start + h
    print(pad_start, pad_end)
    # prior_image_padding = torch.zeros((L, L)).to(device)
    # prior_image_padding[pad_start:pad_end, pad_start:pad_end] = prior_image.to(device)

    train_refine_loader = data.DataLoader(
        dataset=dataset.TrainData_refine_jm(
            view_N=view_N, sinogram=lv_sino, sample_N=sample_N, degrees_all=degrees_all
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = data.DataLoader(dataset=dataset.TestDataDirect(L=L), batch_size=1, shuffle=False)

    l1_loss, l2_loss = torch.nn.SmoothL1Loss(), torch.nn.MSELoss()
    mu = 100
    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for e in tqdm(range(epoch)):
        net.train()
        loss_train, loss_train_fit, loss_train_refine = 0, 0, 0
        for i, (ray_sample, projection_l_sample) in enumerate(train_refine_loader):
            ray_sample = ray_sample.to(device).float().view(-1, 2)
            intensity_sample = F.grid_sample(
                prior_image_padding.unsqueeze(0).unsqueeze(0),
                ray_sample.unsqueeze(0).unsqueeze(0),
                mode="bilinear",
                align_corners=False,
            )[0, 0, 0, :].unsqueeze(-1)
            intensity_pre = net(ray_sample)
            projection_l_sample_pre = (
                torch.sum(intensity_pre.view(-1, sample_N, L, 1), dim=2).squeeze(-1).squeeze(-1)
            )

            loss_fit = l2_loss(intensity_pre, intensity_sample.to(intensity_pre.dtype))
            loss_refine = l1_loss(projection_l_sample_pre, projection_l_sample.to(device).float())

            loss = 0 * loss_refine + mu * loss_fit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            loss_train_fit += loss_fit.item()
            loss_train_refine += loss_refine.item()

        scheduler.step()
        if (e + 1) % 1 == 0:
            print(
                f"(Joint Optim) Epoch[{e + 1}/{epoch}], Lr:{scheduler.get_last_lr()[0]}, Loss:{loss_train / len(train_refine_loader)}, Loss_fit:{loss_train_fit / len(train_refine_loader)}, Loss_refine:{loss_train_refine / len(train_refine_loader)}"
            )

        if (e + 1) % 10 == 0:
            torch.save(net.state_dict(), model_path)
            with torch.no_grad():
                for i, xy in enumerate(test_loader):
                    img_pre_torch = net(xy.to(device).float().view(-1, 2)).view(L, L)
                    img_pre_numpy = (
                        img_pre_torch[pad_start:pad_end, pad_start:pad_end].float().cpu().detach().numpy()
                    )
                    psnr_value = peak_signal_noise_ratio(
                        prior_image_clone.view(256, 256).cpu().numpy(),
                        img_pre_numpy,
                        data_range=prior_image_clone.cpu().numpy().max()
                        - prior_image_clone.cpu().numpy().min(),
                    )
                    print("PSNR:", psnr_value)
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img_pre_numpy), f"{out_path}/Joint_Optim_epoch_{e + 1}.nii"
                    )

    return img_pre_torch.reshape(1, 1, 256, 256)


if __name__ == "__main__":
    gt = sitk.ReadImage(
        "/mnt/raid5/dch/code/DiffusionNerfProj/1_LACT/data/our_pure/L067_fbp_hu_hann_window_256_norm.nii"
    )
    gt = sitk.GetArrayFromImage(gt)[13]

    sitk.WriteImage(sitk.GetImageFromArray(gt), "results/0116/gt.nii")
    sino_gt = radon(gt, theta=np.linspace(0, 90, 360, endpoint=False), circle=False).T

    device = "cuda:1"

    gt_torch = torch.from_numpy(gt).float().to(device)
    sino_gt_torch = torch.from_numpy(sino_gt).float().to(device)

    # train_fit(
    #     config_path="NeRF/config_SVCT.json", prior_image=gt_torch, save_path="results/0115", device="cuda:0"
    # )
    # train_refine(config_path="config_LACT_init.json",
    #              view_N=360,
    #              lv_sino=sino_gt_torch,
    #              degrees_all=90,
    #              save_path="results/0928",
    #              device="cuda:0")
    train_joint_optim(
        config_path="NeRF/config_SVCT.json",
        prior_image=gt_torch,
        view_N=360,
        lv_sino=sino_gt_torch,
        degrees_all=90,
        save_path="results/0116",
        device=device,
    )
