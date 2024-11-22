import argparse
import os
import yaml

import numpy as np
import torch as th

from patch_diffusion import dist_util
from patch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    add_dict_to_dict,
    dict_to_dict,
)
from geometry.build_gemotry import initialization
from geometry.syndeeplesion_data import test_image
import imageio.v2 as imageio
from tqdm import tqdm
from torch_radon import FanBeam

from utils import save_nii_image

# from torch_radon import RadonFanbeam

from torchvision.transforms import Resize

import SimpleITK as sitk

def main(parser):
    args = parser.parse_args()
    yaml_path = args.config

    # parser = argparse.ArgumentParser()
    with open(yaml_path) as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args_dict = add_dict_to_dict(args_dict, model_and_diffusion_defaults())

    a = args_dict["a"]
    n = args_dict["n"]
    delta_y = args_dict["delta_y"]
    save_dir = args_dict["save_dir"]
    data_path = args_dict["data_path"]
    inner_dir = args_dict["inner_dir"]
    n_img = args_dict["num_test_image"]
    n_mask = args_dict["num_test_mask"]


    sino = th.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage('640_sino_pre.nii'))).cuda()

    for imag_idx in range(n_img):  # 200 for all test data
        print(imag_idx)
        for mask_idx in tqdm(range(n_mask)):  # 10
            import matplotlib.pyplot as plt

            Xma, XLI, Xgt, M, Sma, SLI, Sgt, Tr = test_image(data_path, imag_idx, mask_idx, inner_dir)
            print(Xgt.shape, M.shape, Sma.shape, Tr.shape)


    print(sino.shape)

    resize1 = Resize(416)

    fp = lambda x: radon.forward(resize1(x)) * param.reso
    bp = lambda x: resize1(radon.backward(radon.filter_sinogram(x / param.reso)))

    param = initialization()
    angles = np.linspace(-np.pi/2, 3*np.pi/2, 640)
    radon = FanBeam(641, angles, 1075, 1075, param.param["su"] / param.reso / 641)
    print(param.param["su"] / param.reso / 641)
    # radon = RadonFanbeam(641, angles, 1075, 1075, param.param["su"] / param.reso / 641)



    _ = fp(Xgt)  # Must use forward before calling backward or specify a volume
    recon = bp(sino)

    save_nii_image(recon, 'recon.nii')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config/MAR.yaml", help="yaml file for configuration"
    )
    main(parser)
