import os

import numpy as np
import torch
from torchvision import transforms

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class NpyDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.sample_list = os.listdir(self.data_dir)
        self.label_dir = self.data_dir + "_label/"
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        slice_name = self.sample_list[index].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name)
        label_name = slice_name.replace('c001', 'c000').replace('sl0-3', 'sl0-1')
        label_path = self.label_dir + label_name
        image = np.load(data_path)
        label = np.load(label_path)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        transform_img = transforms.Compose([transforms.Resize((self.opt.crop_size, self.opt.crop_size))])
        A = transform_img(image)
        B = transform_img(label)

        return {'A': A, 'B': B, 'A_paths': data_path, 'B_paths': label_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.sample_list)
