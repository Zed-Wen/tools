# DuDoDp-MAR

该目录基于 Xiaokun Liang 老师组的工作 [DuDoDp-MAR](https://github.com/DeepXuan/DuDoDp-MAR)。主要使用其提供的数据、预训练diffusion model等。在此基础上我实现了 DPER、DPS、diffusionMBIR 等方法。

**该目录用于探索 DPER 框架在 CT 金属伪影去除中的应用。**

## 激活环境
``` shell
conda activate /home/dch/.conda/envs/dper-gd/
```

## Test data
Please refer to [SynDeepLesion](https://github.com/hongwang01/SynDeepLesion).

`./test_data` 中存放了 10 个包含不同类型金属伪影的 CT 图像，用于测试。

## 运行 DuDoDp-MAR 方法（只适用于 CT 金属去伪影）
``` python
python mar.py -c config/MAR.yaml
```

## 运行其他方法
以稀疏视角 CT (Sparse View CT) 为例，运行以下命令：
``` python
python sample_uni_lvct.py --task "SVCT" --method "MBIR" --gpu 2 --degree 96 --slice_idx 200
```

在上面的命令中，`--task` 用于指定任务，`--method` 用于指定方法，`--gpu` 用于指定 GPU 编号，`--degree` 用于指定欠采样数据的程度，`--slice_idx` 用于指定测试图像的切片编号。

## 当前目标

- [ ] 用最粗略的 DPER 方法解决金属伪影去除问题


## Acknowledgment
Big thanks to [PatchDiffusion-Pytorch](https://github.com/ericl122333/PatchDiffusion-Pytorch) and [guided-diffusion](https://github.com/openai/guided-diffusion) for providing the codes that facilitated the training of diffusion models, and [SynDeepLesion](https://github.com/hongwang01/SynDeepLesion) for the test data.
