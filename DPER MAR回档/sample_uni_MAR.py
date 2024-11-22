import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    # default="/mnt/raid5/dch/code/DiffusionNerfProj/1_LACT/AAPM_DATA_nii/512/L067.nii.gz",   # 512*512
    default="./test_data/SynDeepLesion",   # 416*416
    dest="data",
    help="Path of sinogram to be recon",
)
parser.add_argument("--task", type=str, default="LACT", dest="task", help="CT recon task")
parser.add_argument("--method", type=str, default="DPER", dest="method", help="Recon method")
parser.add_argument("--gpu", type=int, default=0, dest="gpu", help="ID of GPU")
# parser.add_argument("--slice_idx", type=int, default=30, dest="slice_idx", help="Slice index")
# parser.add_argument("--degree", type=int, default=90, dest="degree", help="Available degree")
parser.add_argument("--dps_weight", type=float, default=0.01, dest="dps_weight", help="DPS weight")
# parser.add_argument("--image_size", type=int, default=416, dest="image_size", help="width of your image data")
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


from tqdm import tqdm
from geometry.syndeeplesion_data import test_image
import matplotlib.pyplot as plt
from utils import save_nii_image
from torchvision.transforms import Resize
from torch_radon import FanBeam
from geometry.build_gemotry import initialization



def main():
    args_dict = yaml.load(open("config/MAR.yaml"), Loader=yaml.FullLoader)
    args_dict = add_dict_to_dict(args_dict, model_and_diffusion_defaults())
    args = parser.parse_args()

    # Load data
    # ------------------------------------------
    data_path = Path(args.data)

    for imag_idx in range(1):  # 200 for all test data
        print('imag_idx:',imag_idx)
        for mask_idx in tqdm(range(1)):  # 10
            inner_dir = "test_640geo/"
            Xma, XLI, Xgt, M, Sma, SLI, Sgt, Tr = test_image(data_path, imag_idx, mask_idx, inner_dir)
            print('Xgt.shape, M.shape, Sma.shape, Tr.shape:',Xgt.shape, M.shape, Sma.shape, Tr.shape)

            Xma_np = Xma.cpu().numpy()   # (1,1,416,416)
            XLI_np = XLI.cpu().numpy()  # (1,1,416,416)
            Xgt_np = Xgt.cpu().numpy()
            M_np= M.cpu().numpy()
            Sma_np = Sma.cpu().numpy()
            SLI_np = SLI.cpu().numpy()
            Sgt_np = Sgt.cpu().numpy() # （1,1,640,641）
            Tr_np = Tr.cpu().numpy()  # （1,1,640,641）
            # Tr_np = 1 - Tr_np

            # Xma_np = (Xma_np - Xma_np.min()) / (Xma_np.max() - Xma_np.min())  # normalize

            fig, axs = plt.subplots(2, 4, figsize=(12, 6))
            axs[0, 0].imshow(Xma_np.squeeze(), cmap="gray")
            axs[0, 0].set_title("Xma")
            axs[0, 1].imshow(XLI_np.squeeze(), cmap="gray")
            axs[0, 1].set_title("XLI")
            axs[0, 2].imshow(Xgt_np.squeeze(), cmap="gray")
            axs[0, 2].set_title("Xgt")
            axs[0, 3].imshow(M_np.squeeze(), cmap="gray")
            axs[0, 3].set_title("M")
            axs[1, 0].imshow(Sma_np.squeeze(), cmap="gray")
            axs[1, 0].set_title("Sma")
            axs[1, 1].imshow(SLI_np.squeeze(), cmap="gray")
            axs[1, 1].set_title("SLI")
            axs[1, 2].imshow(Sgt_np.squeeze(), cmap="gray")
            axs[1, 2].set_title("Sgt")
            axs[1, 3].imshow(Tr_np.squeeze(), cmap="gray")
            axs[1, 3].set_title("Tr")

            plt.tight_layout()
            plt.show()

    # data_image = sitk.GetArrayFromImage(sitk.ReadImage(data_path))[args.slice_idx]
    # data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    view_full_num = 640  # 全视角共720度（存疑，数据上看full view应该是360°,全视角个数是640）！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    if args.task == "LACT":
        return NotImplementedError

    elif args.task == "SVCT":
        return NotImplementedError

    elif args.task == "MAR":
        view_limited_num = view_full_num
        view_limited_idx = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)

    # Create Problem Name
    # ------------------------------------------
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    data_name = args.data.split("/")[-1].split(".")[0]
    problem = f"{args.task}-{view_full_num}-{args.method}-{data_name}-{args.dps_weight}"

    if args.method == "MBIR":
        problem = f"{args.task}-{view_full_num}-{args.method}-{data_name}"

    print(
        f"""当前运行的任务:{problem}
        重建方法:{args.method}
        观测数据:{args.data}
        可视角度:{view_full_num}
        视角数:{len(view_limited_idx)}/{view_full_num}
        DPS权重:{args.dps_weight}"""
    )

    # ---------------------------------------------------------------------------------------------------
    # Generate Radon Transform Operator
    resize1 = Resize(416)  # 用法：resize1(x), 输入x是一个1*1*n*n的cuda tensor，输出为1*1*416*416的缩放后的cuda tensor
    resize2 = Resize(512)
    resize3 = Resize((640,641)) # 用于转换正弦图的size

    param = initialization()
    angles = np.linspace(param.param["startangle"] , param.param["endangle"], param.param["nProj"])
    radon = FanBeam(641, angles, 1075, 1075, param.param["su"] / param.reso / 641)
    fp = lambda x: radon.forward(resize1(x)) * param.reso
    bp = lambda x: resize2(radon.backward(radon.filter_sinogram(x / param.reso)))
    # -----------------------------------------------------------------------------------------------------
    # # 测试fp、bp、radon用
    # device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    # Xgt_th = th.from_numpy(Xgt_np).to(device)

    # radon_torch_mod = ct.CT_Clinical_uni_v2(
    #     img_width=768,
    #     view_limited_idx=view_limited_idx,
    # )

    # pred_xstart_normalization = resize2(Xgt)
    # # print(Sma.shape)    # 1,1,640,641
    # # print(pred_xstart_normalization.shape)
    # # print(radon_torch_mod.A_FV(pred_xstart_normalization).shape)   # 1,1,640,768
    # # plt.imshow(resize3(radon_torch_mod.A_FV(pred_xstart_normalization)).cpu().detach().squeeze(0).squeeze(0))
    # # plt.show()
    # # plt.imshow(Sma.cpu().detach().squeeze(0).squeeze(0))
    # # plt.show()

    # sino_fv = radon_torch_mod.A_FV(Xgt)  # [1, 1, 640, 768]

    # fp_S = fp(Xgt_th)

    # # utils.save_nii_image(sino_fv.squeeze(),  "sino_fv.nii")
    # # utils.save_nii_image(Sgt_np.squeeze(),  "Sgt_np.nii")
    # # utils.save_nii_image(fp_S.squeeze(),  "fp_S.nii")
    # print(sino_fv.squeeze().shape)
    # print(Sgt_np.squeeze().shape)
    # print(fp_S.squeeze().shape)
    # # print(Sgt_np[:3,:3])
    # # print(fp_S.cpu().numpy()[:3,:3])

    # # fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    # # axs[0, 0].imshow(sino_fv.cpu().squeeze(), cmap="gray")
    # # plt.xlim(0, 720)
    # # plt.ylim(0, 768)
    # # axs[0, 0].set_title("sino_fv")
    # # axs[0, 1].imshow(Sgt_np.squeeze(), cmap="gray")
    # # axs[0, 1].set_title("Sgt_np")
    # # plt.xlim(0, 640)
    # # plt.ylim(0, 641)
    # # axs[1, 0].imshow(fp_S.cpu().squeeze(), cmap="gray")
    # # axs[1, 0].set_title("fpS")
    # # plt.xlim(0, 640)
    # # plt.ylim(0, 641)
    # # plt.tight_layout()
    # # plt.show()

    # # print("difference:",np.sum(Sgt_np-fp_S.cpu().numpy()))

    # fbp_fv = radon_torch_mod.FBP_FV(sino_fv) # [1, 1, 768, 768]

    # pad_start = (768 - 512) // 2
    # pad_end = pad_start + 512

    # fbp_fv = fbp_fv[:, :, pad_start:pad_end, pad_start:pad_end]
    # plt.imshow(sino_fv.cpu().squeeze(),cmap='gray')
    # plt.show()
    # plt.imshow(fbp_fv.cpu().squeeze(),cmap='gray')
    # plt.show()


    # S_fp = fp(Xgt_th) 
    # # X_rec = bp(S_fp)
    # X_rec = bp(S_fp)
    # plt.imshow(S_fp.cpu().squeeze(),cmap='gray')
    # plt.show()
    # plt.imshow(X_rec.cpu().squeeze(),cmap='gray')
    # plt.show()


# -------------------------------------------------------------------------------------------------
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

    # data_image = th.from_numpy(data_image).float().unsqueeze(0).unsqueeze(0).to(dist_util.dev())

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 计算Xma经前向再FBP重建得到的fbp_fv（只是为了最后计算PSNR用）
    radon_torch_mod = ct.CT_Clinical_uni_v2(
        img_width=768,
        view_limited_idx=view_limited_idx,
    )

    sino_fv = radon_torch_mod.A_FV(Xma)  # [1, 1, 720, 768]

    fbp_fv = radon_torch_mod.FBP_FV(sino_fv) # [1, 1, 768, 768]

    pad_start = (768 - 512) // 2
    pad_end = pad_start + 512

    fbp_fv = fbp_fv[:, :, pad_start:pad_end, pad_start:pad_end]  # [1, 1, 512, 512])  注：512是刚好把重建结果中没用的黑边去掉。即416*416的图像前向再FBP重建得到的就是768*768，其中的有效区域是512*512

    # fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    # axs[0, 0].imshow(sino_fv.cpu().squeeze(), cmap="gray")
    # axs[0, 0].set_title("sino_fv")
    # axs[0, 1].imshow(fbp_fv.cpu().squeeze(), cmap="gray")
    # axs[0, 1].set_title("fbp_fv")
    # axs[1, 0].imshow(Xgt.cpu().squeeze(), cmap="gray")
    # axs[1, 0].set_title("GT")
    # plt.tight_layout()
    # plt.show()
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # utils.save_nii_image(Xgt.squeeze(), os.path.join(save_root, "GT.nii"))

    # utils.save_nii_image(sino_fv.squeeze(), os.path.join(save_root, "Sino-FV.nii"))

    # utils.save_nii_image(fbp_fv, os.path.join(save_root, "FBP-FV.nii"))

    # ------------------------------------------
    # Run Reconstruction
    print("sampling...")
    model_kwargs = {}

    if args.method == "DPER":
        sample_fn = diffusion.p_dper_mar_loop
        print("sampling with DPER")
    elif args.method == "DPS":
        sample_fn = diffusion.p_dps_mar_loop
        print("sampling with DPS_MAR")
    elif args.method == "MBIR":
        sample_fn = diffusion.p_diffusionMBIR_loop
        print("sampling with MBIR")
    elif args.method == "INR":
        sample_fn = None
        print("Do not use this pipeline.")
        exit(0)

    # sample = sample_fn(
    #     model_fns,  # models
    #     (args.batch_size, 1, args.image_size, args.image_size),  # shape
    #     # (args.batch_size, 1, 416, 416),  # shape
    #     sino=Sma,
    #     view_available=view_limited_idx,
    #     dps_weight=args.dps_weight,
    #     fp=fp,
    #     log_dir=save_root,
    #     clip_denoised=args.clip_denoised,
    #     model_kwargs=model_kwargs,
    #     denoised_fn=denoised_fn,
    #     device=dist_util.dev(),
    #     progress=True,
    #     metal_trace=Tr
    # )

    sample = sample_fn(
        model_fns,
        (args.batch_size, 1, args.image_size, args.image_size),
        sino=Sma,
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
