"""Class to load the data on which we will train the model (contained in model.py)"""

import os

import cv2
import numpy as np
# use Pytorch for faster and more efficient computations
import torch
import torchvision.transforms as transforms
from PIL import Image
# NEWs
from torch.utils.data import DataLoader, Dataset
from options import Options


class HDRDataset(Dataset):
    """
    Custom HDR dataset that returns a dictionary of LDR input image, HDR ground truth image and file path. 
    """

    def __init__(self, mode, opt):
        """
        Build the Dataset instance.
        """

        self.batch_size = opt.batch_size

        if mode == "train":
            self.dataset_path = os.path.join("./dataset/mixed", "train")
        else:
            self.dataset_path = os.path.join("./dataset/mixed", "test")

        self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
        self.hdr_data_path = os.path.join(self.dataset_path, "HDR")

        # paths to LDR and HDR images ->

        self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))
        self.hdr_image_names = sorted(os.listdir(self.hdr_data_path))

    def __getitem__(self, index):
        """
        Get the element at index 'index' in the instance Dataset.
        - tensor of LDR image
        - tensor of HDR image 
        - path of LDR image 
        """
        self.ldr_image_path = os.path.join(
            self.ldr_data_path, self.ldr_image_names[index]
        )

        # transformations on LDR input ->

        ldr_sample = Image.open(self.ldr_image_path).convert("RGB")
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_ldr = transforms.Compose(transform_list)
        ldr_tensor = transform_ldr(ldr_sample)

        # transformations on HDR ground truth ->

        self.hdr_image_path = os.path.join(
            self.hdr_data_path, self.hdr_image_names[index]
        )

        hdr_sample = cv2.imread(self.hdr_image_path, -1).astype(np.float32)

        # transforms.ToTensor() is used for 8-bit [0, 255] range images; can't be used for [0, ∞) HDR images
        # TODO: reshape HDR images from ?? to 256x256
        # TODO: also reshape the LDR images
        transform_list = [
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_hdr = transforms.Compose(transform_list)
        hdr_tensor = transform_hdr(hdr_sample)

        sample_dict = {
            "ldr_image": ldr_tensor, #LDR image in tensor form 
            "hdr_image": hdr_tensor, #HDR image in tensor form 
            "path": self.ldr_image_path, #path of the LDR image
        }

        return sample_dict

    def __len__(self):
        """
        Returns the numbre of LDR images that are taken in one batch.
        """
        return len(self.ldr_image_names) // self.batch_size * self.batch_size
