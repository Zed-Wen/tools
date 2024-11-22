import numpy as np
from torch.utils import data
import torch


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


def build_coordinate(h, w, angle=None):
    step_h = 1 / h
    step_w = 1 / w
    x = np.linspace(-1 + step_h / 2, 1 - step_h / 2, h)
    y = np.linspace(-1 + step_w / 2, 1 - step_w / 2, w)

    x, y = np.meshgrid(x, y, indexing="ij")  # (L, L), (L, L)

    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)

    if angle is not None:
        angle_rad = np.deg2rad(angle)
        trans_matrix = np.array(
            [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
        )
        xy = xy @ trans_matrix.T  # (L*L, 2) @ (2, 2)
        xy = xy.reshape(h, w, 2)
    return xy


def make_coord(shape, ranges=None, flatten=True):
    """
    build coordinate grid
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = 0, 0.5
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    print(ret)
    return ret


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


def build_coordinate_train_fan(xy, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    L, D_2, _ = xy.shape
    xy = xy.reshape(-1, 2)  # (L*2D, 2)
    xy = xy @ trans_matrix.T  # (L*2D, 2) @ (2, 2) -> (L*2D, 2)
    xy = xy.reshape(L, D_2, 2)  # (L, 2D, 2)
    return xy


# ***************************************** #
# * parallel beam
# ***************************************** #
# ======= 1. Prior Embedding =======
class TrainData_fit(data.Dataset):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape
        self.xy = build_coordinate_train(self.h, angle=-90).reshape(-1, 2)
        self.img = self.img.reshape(-1, 1)

    def __getitem__(self, item):
        return self.xy[item], self.img[item]

    def __len__(self):
        return self.xy.shape[0]


class TestData_fit(data.Dataset):
    def __init__(self, img):
        self.h, self.w = img.shape
        self.xy = build_coordinate_train(self.h, angle=-90).reshape(1, int(self.h * self.w), 2)

    def __getitem__(self, item):
        return self.xy[item]  # (h*w, 2)

    def __len__(self):
        return self.xy.shape[0]


# ======= 2. INR Refine =======
# refine train
class TrainData_refine(data.Dataset):
    def __init__(self, view_N, sinogram, sample_N, degrees_all=180):
        self.sample_N = sample_N
        # generate views
        angles = np.linspace(0, degrees_all, view_N, endpoint=False)
        # load limited-view sinogram
        sin = sinogram  # (view, L)
        num_angles, L = sin.shape
        print(sin.shape)
        # store limited-view sinogram and build parallel rays
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            self.rays.append(build_coordinate_train(L=L, angle=angles[i]))

        # self.projections_lines = np.array(self.projections_lines)
        # self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        # sample view
        projection_l = self.projections_lines[item]  # (L, )
        ray = self.rays[item]  # (L, L, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]  # (sample_N, L, 2)
        return ray_sample, projection_l_sample


class TrainData_refine_jm(data.Dataset):
    def __init__(self, view_N, sinogram, sample_N, degrees_all=180):
        self.sample_N = sample_N
        # generate views
        angles = np.linspace(0, degrees_all, view_N, endpoint=False)
        # load limited-view sinogram
        sin = sinogram  # (view, L)
        num_angles, L = sin.shape
        print(sin.shape)
        # store limited-view sinogram and build parallel rays
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            self.rays.append(build_coordinate(h=L, w=L, angle=angles[i]))

        # self.projections_lines = np.array(self.projections_lines)
        # self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        # sample view
        projection_l = self.projections_lines[item]  # (L, )
        ray = self.rays[item]  # (L, L, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]  # (sample_N, L, 2)
        return ray_sample, projection_l_sample


# refine test
class TestDataDirect(data.Dataset):
    def __init__(self, L):
        self.rays = []
        self.rays.append(build_coordinate_train(L=L, angle=-90))

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, L, 2)
        return ray


class TrainFanFit(data.Dataset):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape
        self.xy = fan_coordinate(fan_angle=self.h, D=self.w * 2)  # (L, 2D, 2)
        self.xy = build_coordinate_train_fan(xy=self.xy, angle=-90).reshape(-1, 2)
        self.img = self.img.reshape(-1, 1)

    def __getitem__(self, item):
        return self.xy[item], self.img[item]

    def __len__(self):
        return self.xy.shape[0]


class TestFanFit(data.Dataset):
    def __init__(self, img):
        self.h, self.w = img.shape
        self.xy = fan_coordinate(fan_angle=self.h, D=self.w * 2)  # (L, 2D, 2)
        self.xy = build_coordinate_train_fan(xy=self.xy, angle=-90).reshape(1, int(self.h * self.w), 2)

    def __getitem__(self, item):
        return self.xy[item]  # (h*w, 2)

    def __len__(self):
        return self.xy.shape[0]


class TrainData_fit1(data.Dataset):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape
        self.xy = build_coordinate_train(4 * self.h, angle=-90)
        self.downsamplexy = self.xy[1::4, 1::4, :].reshape(-1, 2)
        self.img = self.img.reshape(-1, 1)

    def __getitem__(self, item):
        return self.downsamplexy[item], self.img[item]

    def __len__(self):
        return self.downsamplexy.shape[0]


class TrainData_fit_PSF(data.Dataset):
    def __init__(self, h):
        self.h = h
        # self.xy = build_coordinate(self.h, self.h).reshape(-1, 2)
        self.xy = make_coord((self.h, self.h))

    def __getitem__(self, item):
        return self.xy[item]

    def __len__(self):
        return self.xy.shape[0]


class TrainDataFitPsf(data.Dataset):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape
        # self.xy = build_coordinate(self.h, self.w).reshape(-1, 2)
        self.xy = make_coord(self.img.shape)
        # print(self.xy.shape)
        self.img = self.img.reshape(-1, 1)

    def __getitem__(self, item):
        return self.xy[item], self.img[item]

    def __len__(self):
        return self.xy.shape[0]


class TestdataFitPsf(data.Dataset):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape

        # self.xy = build_coordinate(self.h * 2, self.w * 2).reshape(
        #     1, int(self.h * 2 * self.w * 2), 2
        # )
        self.xy = make_coord((self.h, self.w))
        print(self.xy.shape)

    def __getitem__(self, item):
        return self.xy[item]

    def __len__(self):
        return self.xy.shape[0]


###### 2. INR Refine ######
class TestData_refine(data.Dataset):
    def __init__(self, theta, L):
        # generate views
        angles = np.linspace(0, 180, theta, endpoint=False)
        num_angles = len(angles)
        # build parallel rays
        self.rays = []
        for i in range(num_angles):
            self.rays.append(build_coordinate_train(L=L, angle=angles[i]))

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, L, 2)
        return ray


# ==================== Fan Beam ====================
class TrainFanRefine(data.Dataset):
    def __init__(self, view_N, sinogram, sample_N, degrees_all=360):
        self.sample_N = sample_N
        # generate views
        angles = np.linspace(0, degrees_all, view_N, endpoint=False)
        # load limited-view sinogram
        sin = sinogram  # (view, L)
        num_angles, L = sin.shape
        print(sin.shape)
        # store limited-view sinogram and build Fan Beam rays
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            xy = fan_coordinate(L, L)  # (L, 2D, 2)
            self.rays.append(build_coordinate_train_fan(xy=xy, angle=angles[i]))

        # self.projections_lines = np.array(self.projections_lines)
        # self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        # sample view
        projection_l = self.projections_lines[item]  # (L, )
        ray = self.rays[item]  # (L, L, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]  # (sample_N, L, 2)
        return ray_sample, projection_l_sample


class TrainClinicalRefine(data.Dataset):
    def __init__(self, view_available, sinogram):
        # load limited-view sinogram
        self.sino = sinogram
        _, _, self.num_angles, self.L = self.sino.shape

    def __len__(self):
        return self.num_angles

    def __getitem__(self, item):
        # sample view
        # sino_view = self.sino[item]
        return item
