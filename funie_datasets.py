# import os
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms
# import random

# def norm(image):
#     return (image / 127.5) - 1.0

# def denorm(image):
#     return (image + 1.0) * 127.5

# def augment(dt_im, eh_im):
#     # Random interpolation
#     a = random.random()
#     dt_im = dt_im * a + eh_im * (1 - a)

#     # Random flip left right
#     if random.random() < 0.25:
#         dt_im = np.fliplr(dt_im)
#         eh_im = np.fliplr(eh_im)

#     # Random flip up down
#     if random.random() < 0.25:
#         dt_im = np.flipud(dt_im)
#         eh_im = np.flipud(eh_im)

#     return dt_im, eh_im

# class PairDataset(torch.utils.data.Dataset):
#     def __init__(self, data_root, im_size):
#         super(PairDataset, self).__init__()

#         self.data_root = data_root
#         self.im_size = im_size

#         # Get the list of image filenames in trainA and trainB folders
#         self.dt_ims = [os.path.join(data_root, "trainA", fname) for fname in os.listdir(os.path.join(data_root, "trainA"))]
#         self.eh_ims = [os.path.join(data_root, "trainB", fname) for fname in os.listdir(os.path.join(data_root, "trainB"))]
#         print(f"Total {len(self.dt_ims)} data")

#     def __getitem__(self, index):
#         # Read and resize image pair
#         # dt_im = Image.open(self.dt_ims[index]).convert("RGB").resize(self.im_size)
#         # eh_im = Image.open(self.eh_ims[index]).convert("RGB").resize(self.im_size)
#         dt_im = Image.open(self.dt_ims[index]).convert("RGB")
#         eh_im = Image.open(self.eh_ims[index]).convert("RGB")
#         dt_im = dt_im.resize(self.im_size)
#         eh_im = eh_im.resize(self.im_size)
        

#         # Convert image pair to float32 np.ndarray
#         dt_im = np.array(dt_im, dtype=np.float32)
#         eh_im = np.array(eh_im, dtype=np.float32)

#         # Transform image pair
#         dt_im, eh_im = self.transform(dt_im, eh_im)

#          # Augment image pair
#         dt_im, eh_im = augment(dt_im, eh_im)

#         # Transfrom image pair to (C, H, W) torch.Tensor
#         dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
#         eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
#         return dt_im, eh_im

# # Read and resize image pair
# #         dt_im = Image.open(self.dt_ims[index]).convert("RGB")
# #         eh_im = Image.open(self.eh_ims[index]).convert("RGB")
# #         dt_im = dt_im.resize(self.im_size)
# #         eh_im = eh_im.resize(self.im_size)

# #         # Transfrom image pair to float32 np.ndarray
# #         dt_im = np.array(dt_im, dtype=np.float32)
# #         eh_im = np.array(eh_im, dtype=np.float32)

# #         # Augment image pair
# #         if self.split == "train":
# #             dt_im, eh_im = augment(dt_im, eh_im)

# #         # Transfrom image pair to (C, H, W) torch.Tensor
# #         dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
# #         eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
# #         return dt_im, eh_im

#     def transform(self, dt_im, eh_im):
#         # Augment image pair if needed
#         # You can add your augmentation logic here if required
#         # For now, just normalize the images

#         # Normalize image pair
#         dt_im = (dt_im / 255.0 - 0.5) * 2.0
#         eh_im = (eh_im / 255.0 - 0.5) * 2.0

#         return dt_im, eh_im

#     def __len__(self):
#         return len(self.dt_ims)

# if __name__ == "__main__":
#     dataset = PairDataset(data_root="/home/undrdawatr/Vision/Project/MiddleDataset/", im_size=(256, 256))
#     image, target = dataset[0]






# ------------------------------------------------------------------------------------------------------------------------------------------------------#
import os
from torchvision import transforms
import glob
import json
import random

import numpy as np
import torch
from PIL import Image


def norm(image):
    return (image / 127.5) - 1.0

def denorm(image):
    return (image + 1.0) * 127.5

def augment(dt_im, eh_im):
    # Random interpolation
    a = random.random()
    dt_im = dt_im * a + eh_im * (1 - a)

    # Random flip left right
    if random.random() < 0.25:
        dt_im = np.fliplr(dt_im)
        eh_im = np.fliplr(eh_im)

    # Random flip up down
    if random.random() < 0.25:
        dt_im = np.flipud(dt_im)
        eh_im = np.flipud(eh_im)

    return dt_im, eh_im


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        # super(PairDataset, self).__init__()

        # self.data_root = data_root
        # self.im_size = im_size
        # self.split = split

        # # Load JSON of splits
        # names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]

        # # Build image paths
        # self.dt_ims = [f"{self.data_root}/trainA/{n}" for n in names]
        # self.eh_ims = [f"{self.data_root}/trainB/{n}" for n in names]
        # print(f"Total {len(self.dt_ims)} data")

        super(PairDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split
        
        # Get the list of image filenames in trainA and trainB folders
        self.dt_ims = [os.path.join(data_root, "trainA", fname) for fname in os.listdir(os.path.join(data_root, "trainA"))]
        self.eh_ims = [os.path.join(data_root, "trainB", fname) for fname in os.listdir(os.path.join(data_root, "trainB"))]
        print(f"Total {len(self.dt_ims)} data")


    def __getitem__(self, index):
        # Read and resize image pair
        dt_im = Image.open(self.dt_ims[index]).convert("RGB")
        eh_im = Image.open(self.eh_ims[index]).convert("RGB")
        dt_im = dt_im.resize(self.im_size)
        eh_im = eh_im.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt_im = np.array(dt_im, dtype=np.float32)
        eh_im = np.array(eh_im, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt_im, eh_im = augment(dt_im, eh_im)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
        eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
        return dt_im, eh_im

    def __len__(self):
        return len(self.dt_ims)


class UnpairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        super(UnpairDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split

        # Load JSON of splits
        names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]

        # Build image paths
        self.dt_ims = [f"{self.data_root}/{n}" for n in names if "trainA" in n]
        self.eh_ims = [f"{self.data_root}/{n}" for n in names if "trainB" in n]
        print(f"Total {len(self.dt_ims)} poor quality data")
        print(f"Total {len(self.eh_ims)} good quality data")

        # Force # of images to the least amount
        num = min(len(self.dt_ims), len(self.eh_ims))
        self.dt_ims = self.dt_ims[:num]
        self.eh_ims = self.eh_ims[:num]
        print(f"Total {len(self.eh_ims)} data used")

    def __getitem__(self, index):
        # Read and resize image pair
        dt_im = Image.open(self.dt_ims[index]).convert("RGB")
        eh_im = Image.open(self.eh_ims[index]).convert("RGB")
        dt_im = dt_im.resize(self.im_size)
        eh_im = eh_im.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt_im = np.array(dt_im, dtype=np.float32)
        eh_im = np.array(eh_im, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt_im, eh_im = augment(dt_im, eh_im)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
        eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
        return dt_im, eh_im

    def __len__(self):
        return len(self.dt_ims)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size):
        super(TestDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.ims = glob.glob(f"{self.data_root}/*")

    def __getitem__(self, index):
        # Read and resize image
        path = self.ims[index]
        im = Image.open(path).convert("RGB")
        im = im.resize(self.im_size)

        # Transfrom image to float32 np.ndarray
        im = np.array(im, dtype=np.float32)

        # Transfrom image to (C, H, W) torch.Tensor
        im = torch.Tensor(norm(im)).permute(2, 0, 1)
        
        return path, im

    def __len__(self):
        return len(self.ims)


if __name__ == "__main__":
    dataset = PairDataset(
        # data_root="../data/EUVP Dataset/Paired/underwater_dark", im_size=(256, 256))
        data_root="./saved_model/MiddleDataset/", im_size=(256, 256))
    image, target = dataset[0]
