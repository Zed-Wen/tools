import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    default="/mnt/raid5/dch/code/DiffusionNerfProj/1_LACT/AAPM_DATA_nii/512/L067.nii.gz",
    dest="data",
    help="Path of sinogram to be recon",
)
parser.add_argument("--task", type=str, default="LACT", dest="task", help="CT recon task")
parser.add_argument("--method", type=str, default="DPER", dest="method", help="Recon method")
parser.add_argument("--gpu", type=int, default=0, dest="gpu", help="ID of GPU")
parser.add_argument("--slice_idx", type=int, default=30, dest="slice_idx", help="Slice index")
parser.add_argument("--degree", type=int, default=90, dest="degree", help="Available degree")
parser.add_argument("--dps_weight", type=float, default=0.01, dest="dps_weight", help="DPS weight")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch as th
from patch_diffusion import dist_util
from patch_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    add_dict_to_dict,
    dict_to_dict,
)
from pathlib import Path
import utils
import datetime
import ct
import SimpleITK as sitk


def main():
    args_dict = yaml.load(open("config/MAR.yaml"), Loader=yaml.FullLoader)
    args_dict = add_dict_to_dict(args_dict, model_and_diffusion_defaults())
    args = parser.parse_args()

    # Load data
    # ------------------------------------------
    data_path = Path(args.data)
    data_image = sitk.GetArrayFromImage(sitk.ReadImage(data_path))[args.slice_idx]
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    view_full_num = 720 

    if args.task == "LACT":
        view_limited_num = int(view_full_num * (args.degree / 360))
        view_limited_idx = np.linspace(0, view_limited_num, view_limited_num, endpoint=False, dtype=int)

    elif args.task == "SVCT":
        view_limited_num = args.degree
        view_limited_idx = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)

    # Create Problem Name
    # ------------------------------------------
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    data_name = args.data.split("/")[-1].split(".")[0]
    problem = f"{args.task}-{args.degree}-{args.method}-{data_name}-{args.slice_idx}-{args.dps_weight}"

    if args.method == "MBIR":
        problem = f"{args.task}-{args.degree}-{args.method}-{data_name}-{args.slice_idx}"

    print(
        f"""当前运行的任务:{problem}
        重建方法:{args.method}
        观测数据:{args.data}
        可视角度:{args.degree}
        视角数:{len(view_limited_idx)}/{view_full_num}
        DPS权重:{args.dps_weight}"""
    )

    # Create Save Path
    # ------------------------------------------
    save_root = Path(f"./experiments/sample/{args.method}/{problem}/{time_str}/")
    save_root.mkdir(parents=True, exist_ok=True)

    # Setup Diffusion Model
    # ------------------------------------------
    print("creating model and diffusion...")
    model_names = args_dict["model_path"].split(",")
    print("model names: ", model_names)
    models = []

    for model_name in model_names:
        model, diffusion = create_model_and_diffusion(
            **dict_to_dict(args_dict, model_and_diffusion_defaults().keys())
        )
        add_dict_to_argparser(parser, args_dict)
        args = parser.parse_args()
        model.load_state_dict(dist_util.load_state_dict(model_name, map_location="cpu"))
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        models.append(model)

    model_fns = models
    denoised_fn = None

    # Generate Radon Transform Operator
    # ------------------------------------------
    data_image = th.from_numpy(data_image).float().unsqueeze(0).unsqueeze(0).to(dist_util.dev())

    radon_torch_mod = ct.CT_Clinical_uni_v2(
        img_width=768,
        view_limited_idx=view_limited_idx,
    )
    sino_lv = radon_torch_mod.A_LV(data_image).clone().detach()
    sino_fv = radon_torch_mod.A_FV(data_image)

    fbp_lv = radon_torch_mod.FBP_LV(sino_lv)
    fbp_fv = radon_torch_mod.FBP_FV(sino_fv)

    pad_start = (768 - 512) // 2
    pad_end = pad_start + 512

    fbp_lv = fbp_lv[:, :, pad_start:pad_end, pad_start:pad_end]
    fbp_fv = fbp_fv[:, :, pad_start:pad_end, pad_start:pad_end]

    utils.save_nii_image(data_image.squeeze(), os.path.join(save_root, "GT.nii"))

    utils.save_nii_image(sino_lv.squeeze(), os.path.join(save_root, "Sino-LV.nii"))
    utils.save_nii_image(sino_fv.squeeze(), os.path.join(save_root, "Sino-FV.nii"))

    utils.save_nii_image(fbp_lv, os.path.join(save_root, f"FBP-LV.nii"))
    utils.save_nii_image(fbp_fv, os.path.join(save_root, "FBP-FV.nii"))

    # ------------------------------------------
    # Run Reconstruction
    print("sampling...")
    model_kwargs = {}

    if args.method == "DPER":
        sample_fn = diffusion.p_dper_loop
        print("sampling with DPER")
    elif args.method == "DPS":
        sample_fn = diffusion.p_dps_loop
        print("sampling with DPS")
    elif args.method == "MBIR":
        sample_fn = diffusion.p_diffusionMBIR_loop
        print("sampling with MBIR")

    sample = sample_fn(
        model_fns,
        (args.batch_size, 1, args.image_size, args.image_size),
        sino=sino_lv,
        view_available=view_limited_idx,
        dps_weight=args.dps_weight,
        radon=radon_torch_mod,
        log_dir=save_root,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        denoised_fn=denoised_fn,
        device=dist_util.dev(),
        progress=True,
    )

    print("sampling complete")

    sample = sample + 1
    utils.save_nii_image(sample, os.path.join(save_root, "recon.nii"))

    psnr, ssim = utils.cal_metrics(sample, fbp_fv, os.path.join(save_root, "recon.nii"))
    print(f"PSNR: {psnr}, SSIM: {ssim}")


if __name__ == "__main__":
    main()
