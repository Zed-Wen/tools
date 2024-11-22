import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import commentjson as json
import numpy as np
import tinycudann as tcnn
import torch
import pathlib
from torch.optim import lr_scheduler
from tqdm import tqdm
import tifffile
import yaml

import torch_radon
import NeRF.NeRF_dataset as dataset
from torchmetrics.image import PeakSignalNoiseRatio
from utils import save_nii_image
import SimpleITK as sitk
from pathlib import Path
from torch.utils import data
from ct import CT_Clinical, CT_Clinical_uni


def load_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def initialize_model_and_optimizer(config, device):
    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["train_fit"]["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config["train_fit"]["lr_decay_epoch"],
        gamma=config["train_fit"]["lr_decay_coefficient"],
    )
    return net, optimizer, scheduler


def log_progress(epoch, loss, scheduler, psnr_value=None):
    log_msg = f"Epoch[{epoch}], Lr:{scheduler.get_last_lr()[0]}, Loss:{loss:.6f}"
    if psnr_value is not None:
        log_msg += f", PSNR: {psnr_value}"
    print(log_msg)


def train_fit(config_path, prior_image, save_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    h, w = config["file"]["h"], config["file"]["w"]
    recon_size = config["file"]["recon_size"]

    train_params = config["train_fit"]
    device = prior_image.device
    out_path = os.path.join(save_path, "NeRF")
    model_save_path = os.path.join(save_path, "NeRF_checkpoint")

    pad_start = (h - recon_size) // 2
    pad_end = pad_start + recon_size

    for path in [out_path, model_save_path]:
        os.makedirs(path, exist_ok=True)

    coor = torch.from_numpy(dataset.build_coordinate(h, w)).to(device)

    l2_loss = torch.nn.MSELoss()
    psnr = PeakSignalNoiseRatio(data_range=prior_image.max() - prior_image.min()).to(device)

    net, optimizer, scheduler = initialize_model_and_optimizer(config, device)

    for e in tqdm(range(train_params["epoch"])):
        net.train()
        img_pre = net(coor).view(1, 1, h, w).float()

        pixel_wise_loss = l2_loss(img_pre, prior_image)

        loss = pixel_wise_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        avg_loss = loss.item()

        psnr_value = psnr(
            prior_image[:, :, pad_start:pad_end, pad_start:pad_end],
            img_pre[:, :, pad_start:pad_end, pad_start:pad_end],
        ).item()

        if (e + 1) % 10 == 0:
            print(
                f"(1. Prior Embedding) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.8f}, PSNR:{psnr_value:.4f}"
            )

        if psnr_value >= 45.0:
            torch.save(net.state_dict(), os.path.join(model_save_path, "Prior_Embedding.pkl"))
            break
        if (e + 1) % train_params["summary_epoch"] == 0 or e == 0:
            with torch.no_grad():
                img_pre = net(coor).view(1, 1, h, w).float()
                psnr_value = psnr(
                    prior_image[:, :, pad_start:pad_end, pad_start:pad_end],
                    img_pre[:, :, pad_start:pad_end, pad_start:pad_end],
                ).item()
                print("PSNR:", psnr_value)
                img_pre = img_pre[:, :, pad_start:pad_end, pad_start:pad_end]
                save_nii_image(img_pre, os.path.join(out_path, f"Prior_Embedding_epoch_{e + 1}.nii"))
                torch.save(net.state_dict(), os.path.join(model_save_path, "Prior_Embedding.pkl"))

    return img_pre


def train_refine(config_path, sino, view_available, radon, save_path, metal_trace, init=False):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    h, w = config["file"]["h"], config["file"]["w"]
    recon_size = config["file"]["recon_size"]
    train_params = config["train_refine"]
    print(sino.shape)

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")
    model_path = os.path.join(model_save_path, "Prior_Embedding.pkl")

    for path in [out_path, model_save_path]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    device = sino.device

    coor = torch.from_numpy(dataset.build_coordinate(h, w)).to(device)

    mask = metal_trace
    smooth_l1_loss = torch.nn.SmoothL1Loss()
    # l2_loss = torch.nn.MSELoss()

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)

    # train_refine_loader = data.DataLoader(
    #     dataset=dataset.TrainClinicalRefine(view_available=view_available, sinogram=sino),
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=0,
    # )

    if not init:
        net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=train_params["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=train_params["lr_decay_epoch"], gamma=train_params["lr_decay_coefficient"]
    )
    sino = sino.squeeze()
    for e in tqdm(range(train_params["epoch"])):
        net.train()
        loss_train = 0
        # for i, (item) in enumerate(train_refine_loader):
        pre_image = net(coor).view(1, 1, h, w).float()

        sino_pred = radon.A_LV(pre_image).squeeze()

        pixel_wise_loss = smooth_l1_loss(sino, sino_pred)
        print(pixel_wise_loss.shape)
        exit(0)

        loss = pixel_wise_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        scheduler.step()
        avg_loss = loss_train

        if (e + 1) % 10 == 0:
            print(
                f"(2. INR Refine) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.6f}"
            )

        if (e + 1) % train_params["summary_epoch"] == 0:
            with torch.no_grad():
                img_pre = net(coor).view(1, 1, h, w).float()
                sino_pred = radon.A_FV(img_pre)
                # sino_pred[:, :, view_available, :] = sino[None, None, view_available, :]
                img_pre = radon.FBP_FV(sino_pred, filter_name="hann")
                pad_start = (h - recon_size) // 2
                pad_end = pad_start + recon_size
                img_pre = img_pre[:, :, pad_start:pad_end, pad_start:pad_end]
                save_nii_image(img_pre, os.path.join(out_path, f"img_pre_{e + 1}.nii"))

    return img_pre


def train_one_step(config_path, prior_image, sino, radon, save_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    h, w = config["file"]["h"], config["file"]["w"]
    recon_size = config["file"]["recon_size"]
    train_params = config["train_refine"]
    print(sino.shape)

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")

    for path in [out_path, model_save_path]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    device = sino.device

    coor = torch.from_numpy(dataset.build_coordinate(h, w)).to(device)

    smooth_l1_loss = torch.nn.SmoothL1Loss()
    l2_loss = torch.nn.MSELoss()
    psnr = PeakSignalNoiseRatio(data_range=prior_image.max() - prior_image.min()).to(device)

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=train_params["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    sino = sino.squeeze()
    for e in tqdm(range(train_params["epoch"])):
        net.train()
        loss_train = 0

        img_pre = net(coor).view(1, 1, h, w).float()

        prior_consistancy_loss = l2_loss(img_pre, prior_image)

        sino_pred = radon.A_LV(img_pre).squeeze()
        data_fidelity_loss = smooth_l1_loss(sino, sino_pred)

        fidelity_weight = 1
        loss = prior_consistancy_loss + data_fidelity_loss * fidelity_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        scheduler.step()
        avg_loss = loss_train

        if (e + 1) % 10 == 0:
            print(
                f"(2. INR Refine) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.6f}"
            )

        if (e + 1) % train_params["summary_epoch"] == 0:
            with torch.no_grad():
                img_pre = net(coor).view(1, 1, h, w).float()
                sino_pred = radon.A_FV(img_pre)
                img_pre = radon.FBP_FV(sino_pred, filter_name="hann")
                pad_start = (h - recon_size) // 2
                pad_end = pad_start + recon_size
                img_pre = img_pre[:, :, pad_start:pad_end, pad_start:pad_end]
                save_nii_image(img_pre, os.path.join(out_path, f"img_pre_{e + 1}.nii"))

    return img_pre


def train_refine_simulation(config_path, sino, radon, save_path, init=False):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    h, w = config["file"]["h"], config["file"]["w"]
    train_params = config["train_refine"]
    print(sino.shape)

    out_path, model_save_path = os.path.join(save_path, "NeRF"), os.path.join(save_path, "NeRF_checkpoint")
    model_path = os.path.join(model_save_path, "Prior_Embedding.pkl")

    for path in [out_path, model_save_path]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    device = sino.device

    coor = torch.from_numpy(dataset.build_coordinate(h, w)).to(device)

    smooth_l1_loss = torch.nn.SmoothL1Loss()
    l2_loss = torch.nn.MSELoss()

    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"], network_config=config["network"]
    ).to(device)

    train_refine_loader = data.DataLoader(
        dataset=dataset.TrainClinicalRefine(view_available=view_available, sinogram=sino),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    if not init:
        net.load_state_dict(torch.load("./experiments/0420/NeRF_checkpoint/Prior_Embedding.pkl"))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=train_params["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    sino = sino.squeeze()
    for e in tqdm(range(train_params["epoch"])):
        net.train()
        loss_train = 0
        for i, (item) in enumerate(train_refine_loader):

            pre_image = net(coor).view(1, 1, h, w).float()

            sino_pred = radon.A_LV(pre_image).squeeze()
            l2_loss2 = smooth_l1_loss(sino_pred[item], sino[item])
            # ssim_loss = 1 - ms_ssim_loss(
            #     sino[None, None, :view_available, :], sino_pred[None, None, :view_available, :]
            # )
            ssim_loss = 0
            lamda = 1
            loss = l2_loss2 * lamda + ssim_loss * (1 - lamda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()
        avg_loss = loss_train

        if (e + 1) % 10 == 0:
            print(
                f"(2. INR Refine) Epoch[{e + 1}/{train_params['epoch']}], Lr:{scheduler.get_last_lr()[0]}, Loss:{avg_loss:.8f}"
            )

        if (e + 1) % train_params["summary_epoch"] == 0:
            with torch.no_grad():
                img_pre = net(coor).view(1, 1, h, w).float()
                # sino_pre = radon_torch_mod.forward(img_pre) * 10
                # # sino_pre[:, :, :view_available, :] = sino[None, None, :view_available, :]
                # filtered_sinogram = radon_torch_mod.filter_sinogram(sino_pre)
                # fbp_pre = radon_torch_mod.backprojection(filtered_sinogram)
                # sino_zero = torch.zeros_like(sino)
                # sino_zero[0:-1:sparse_scale] = sino[0:-1:sparse_scale]
                # filtered_sino = radon_torch_mod.filter_sinogram(sino_zero[None, None, :, :])
                # fbp = radon_torch_mod.backprojection(filtered_sino)
                # save_nii_image(fbp, os.path.join(out_path, f"fbp_{e + 1}.nii"))
                save_nii_image(img_pre, os.path.join(out_path, f"img_pre_{e + 1}.nii"))

                # save_nii_image(fbp_pre, os.path.join(out_path, f"INR_Refine_epoch_{e + 1}.nii"))
                # save_nii_image(sino_zero, os.path.join(out_path, f"INR_Refine_sino_{e + 1}.nii"))
                # save_nii_image(
                #     sino_pred,
                #     os.path.join(out_path, f"INR_Refine_sino_pred_{e + 1}.nii"),
                # )

    return img_pre


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


if __name__ == "__main__":
    device = "cuda"

    sino_path = f"/mnt/raid5/dch/code/DiffusionNerfProj/1_LACT/data/our_pure/helix2fan/out/L004/L004_flat_fan_projections.tif"
    projections, meta_data = load_tiff_stack_with_metadata(Path(sino_path))
    projections = np.copy(np.flip(projections[:, :, 30], axis=1))
    print("Measurement shape: ", projections.shape)
    view_full_num, det_count = projections.shape

    view_available = int(view_full_num * (90 / 360))
    print("Available view number: ", view_available)
    
    view_limited_num = 150
    view_limited_idx = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)

    voxel_size = 0.6
    vox_scaling = 1 / voxel_size

    projections = torch.from_numpy(projections).float().unsqueeze(0).unsqueeze(0).to(device)
    projections = projections * vox_scaling * 20
    measurement = projections[:, :, view_limited_idx, :]

    print(measurement.shape)

    radon_torch_mod = CT_Clinical_uni(
        meta_data=meta_data,
        img_width=int(meta_data["dsd"]),
        view_limited_idx=view_limited_idx,
        det_count=det_count,
        voxel_size=voxel_size,
    )
    fbp_la = radon_torch_mod.FBP_LV(measurement, filter_name="hann")
    fbp_all = radon_torch_mod.FBP_FV(projections, filter_name="hann")

    pad_start = (int(meta_data["dsd"]) - 416) // 2
    pad_end = pad_start + 416

    fbp_la = fbp_la[:, :, pad_start:pad_end, pad_start:pad_end]
    fbp_all = fbp_all[:, :, pad_start:pad_end, pad_start:pad_end]

    save_nii_image(fbp_la, os.path.join("./experiments", "fbp_la.nii"))
    save_nii_image(fbp_all, os.path.join("./experiments", "fbp_all.nii"))

    # prior_image_path = f"./experiments/sample/openai-2024-04-18-15-58-26-549847/x0_200.nii"

    # prior_image = sitk.GetArrayFromImage(sitk.ReadImage(prior_image_path))
    # print(prior_image.shape)
    # prior_image = prior_image + 1
    # prior_image = torch.from_numpy(prior_image).float().to(device).view(1, 1, 416, 416)
    # prior_image_padding = torch.zeros(1, 1, int(meta_data["dsd"]), int(meta_data["dsd"])).to(device)
    # prior_image_padding[:, :, pad_start:pad_end, pad_start:pad_end] = prior_image

    # gt_torch = torch.from_numpy(gt).float().to(device).view(1, 1, 416, 416)
    fbp_all_norm = (fbp_all - fbp_all.min()) / (fbp_all.max() - fbp_all.min())
    measurement_simu = radon_torch_mod.A_LV(fbp_all_norm)

    # train_fit(
    #     config_path="./NeRF/config_LACT_clinical.yaml",
    #     prior_image=prior_image_padding,
    #     save_path="./experiments/0421",
    # )

    # train_refine_simulation(
    #     config_path="./NeRF/config_LACT_clinical.yaml",
    #     sino=measurement_simu,
    #     radon=radon_torch_mod,
    #     save_path="./experiments/0421-simu",
    #     init=False,
    # )
    # exit(0)

    # train_fit_clinical(
    #     config_path="./NeRF/config_LACT_clinical.yaml",
    #     prior_image=prior_image_padding,
    #     save_path="./experiments/0420",
    # )
    train_refine(
        config_path="./NeRF/config_SVCT_clinical.yaml",
        sino=measurement,
        radon=radon_torch_mod,
        save_path="./experiments/0425-hash",
        init=True,
        view_available=view_limited_idx,
    )

    # train_joint_optim(
    #     config_path="NeRF/config_SVCT.json",
    #     prior_image=gt_torch,
    #     view_N=360,
    #     lv_sino=sino_gt_torch,
    #     degrees_all=90,
    #     save_path="results/0116",
    #     device=device,
    # )
