import h5py

import matplotlib.pyplot as plt
import numpy as np
from utils import save_nii_image
# Open the file as read only
# for i in range(1, 10):
#     file = h5py.File(f"test_data/SynDeepLesion/test_640geo/000365_03_01/473/{i}.h5", "r")

#     print("Keys: %s" % file.keys())

#     image = file['ma_CT']
#     print(image.shape)
#     plt.figure()
#     plt.imshow(image, cmap='gray')
#     plt.savefig(f'test_data/SynDeepLesion/test_640geo/000365_03_01/473/{i}.png')

file = h5py.File("test_data/SynDeepLesion/test_640geo/000365_03_01/473/gt.h5", "r")

print("Keys: %s" % file.keys())

for i in file.keys():
    print(i)
    image = file[i]
    print(image.shape)
    save_nii_image(image, f"test_data/SynDeepLesion/test_640geo/000365_03_01/473/{i}.nii.gz")


# # load npy
# image = np.load("test_data/SynDeepLesion/testmask.npy")
# print(image.shape)

# save_nii_image(image, "test_data/SynDeepLesion/testmask.nii.gz")