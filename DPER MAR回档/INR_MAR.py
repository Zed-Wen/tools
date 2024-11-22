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

import SimpleITK as sitk, numpy as np, tinycudann as tcnn, commentjson as json
from torch.optim import lr_scheduler

def setting_network():
    # Parameter setting
    encoding = {"otype": "Grid", 
                "type": "Hash", 
                "n_levels": 8, 
                "n_features_per_level": 8, 
                "log2_hashmap_size": 24, 
                "base_resolution": 2, 
                "per_level_scale": 1.95, 
                "interpolation": "Linear"}
    network = {"otype": "FullyFusedMLP", 
                "activation": "ReLU", 
                "output_activation": "Sigmoid", 
                "n_neurons": 64, 
                "n_hidden_layers": 2}
    return encoding, network

def setting_training_parameter():
    # Parameter setting
    train = {
        "lr": 1e-3,
        "epoch": 5000,
        "summary_epoch": 500,
        "sample_N": 10,
        "batch_size": 3
    },
    return train

def get_INR_network():
    myencoding, mynetwork = setting_network()
    InrNetwork = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, encoding_config=myencoding, network_config=mynetwork)     # 这句是hash encoding + 引入那个3层MLP 二合一
    return InrNetwork



def main():
    args_dict = yaml.load(open("config/MAR.yaml"), Loader=yaml.FullLoader)
    args = parser.parse_args()

    train_param = setting_training_parameter()
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

    view_full_num = 720  # 全视角共720度

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
    angles = np.linspace(-np.pi / 2 * 1, np.pi / 2 * 3, 640)
    param = initialization()
    radon = FanBeam(641, angles, 1075, 1075, param.param["su"] / param.reso / 641)
    fp = lambda x: radon.forward(resize1(x)) * param.reso
    bp = lambda x: resize2(radon.backward(radon.filter_sinogram(x / param.reso)))
    # -----------------------------------------------------------------------------------------------------

    # Create Save Path
    # ------------------------------------------
    save_root = Path(f"./experiments/sample/{args.method}/{problem}/{time_str}/")
    save_root.mkdir(parents=True, exist_ok=True)

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
    # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # creating model
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    SCOPE = get_INR_network()
    l1_loss_function = th.nn.L1Loss()
    optimizer = th.optim.Adam(params=(SCOPE.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    #     train = {
    #     "lr": 1e-3,
    #     "epoch": 5000,
    #     "summary_epoch": 500,
    #     "sample_N": 10,
    #     "batch_size": 3
    # },
    epoch = train_param['epoch']
    lr = train_param['lr']
    summary_epoch = train_param['summary_epoch']
    
    for e in range(epoch):
        SCOPE.train()
        loss_train = 0
        for i, (ray_sample, projection_l_sample) in enumerate(train_loader):
            ray_sample = ray_sample.to(DEVICE).float().view(-1, 2)  # view完之后，每一行都是一个坐标
            projection_l_sample = projection_l_sample.to(DEVICE).float()
            pre_intensity = SCOPE(ray_sample).view(batch_size, sample_N, L, 1) # 3*10*367*1，10是采样了10根射线的结果，全保留的话应该是367*367的

            projection_l_sample_pre = th.sum(pre_intensity, dim=2)  # 3*10*1 根据学出的object得到的投影结果

            projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1) # 3*10 (其实一个squeeze就够了)

            loss = l1_loss_function(projection_l_sample_pre, projection_l_sample.to(projection_l_sample_pre.dtype))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        else:
            scheduler.step()
            print('{}, (TRAIN0) Epoch[{}/{}], Lr:{}, Loss:{:.6f}'.format(num_sv, e + 1, epoch, scheduler.get_last_lr()[0], loss_train / len(train_loader)))

        if (e + 1) % summary_epoch == 0:
            th.save(SCOPE.state_dict(), model_path)


    # ------------------------------------------
    # Run Reconstruction
    if args.method == "INR":
        print("Using INR")
    else:
        print("Using sample_uni")
        exit(0)


    utils.save_nii_image(sample, os.path.join(save_root, "recon.nii"))

    psnr, ssim = utils.cal_metrics(sample, fbp_fv, os.path.join(save_root, "recon.nii"))
    print(f"PSNR: {psnr}, SSIM: {ssim}")


if __name__ == "__main__":
    main()
