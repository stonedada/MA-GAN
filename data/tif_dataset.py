import os
import numpy as np
import tifffile
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
import cv2


class TifDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def mask(self):
        GrayImage = cv2.imread(self.gt_path, 0)
        # 通过阈值抑制，来去除背景部分
        ret, thresh = cv2.threshold(GrayImage, 20, 255, cv2.THRESH_BINARY)
        if ret:
            kernel = np.ones((6, 6), dtype=np.uint8)
            dilate = cv2.dilate(thresh, kernel, 3)  # 1:迭代次数，也就是执行几次膨胀操作
            return np.array(dilate, dtype=np.uint16)
        else:
            assert 'can\'t generate a mask from GrayImage'

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.image_dir = os.path.join(self.data_dir, "npz")
        self.image_list = os.listdir(self.image_dir)
        self.label_dir = os.path.join(self.data_dir, "npz_label")
        self.mask_dir = os.path.join(self.data_dir, "npz_mask")
        # self.gt_dir = os.path.join(opt.gtroot, opt.phase)
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
            C (tensor) - - its corresponding image in th  seg_GT
            image_paths (str) - - image paths
        """

        slice_name = self.image_list[index].strip('\n')
        image_path = os.path.join(self.image_dir, slice_name)
        label_name = slice_name.replace('srs', 'fluo')
        label_path = os.path.join(self.label_dir, label_name)
        # gt_name = label_name.replace('tif', 'png')
        # self.gt_path = os.path.join(self.gt_dir, gt_name)
        mask_name = label_name.replace("png", "tif")
        mask_path = os.path.join(self.mask_dir, mask_name)
        # read a image given a random integer index
        # A = tifffile.imread(image_path)
        # B = tifffile.imread(label_path)
        A = cv2.imread(image_path, -1)
        B = cv2.imread(label_path, -1)
        C = tifffile.imread(mask_path)

        # Transfer numpy to PIL
        A = Image.fromarray(A)
        B = Image.fromarray(B)
        C = Image.fromarray(C)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        Image_transform = get_transform(self.opt, transform_params, grayscale=True)

        # A = A_transform(A)
        A = Image_transform(A)
        A = A.repeat(3, 1, 1)
        B = Image_transform(B)
        C = Image_transform(C)

        return {'A': A, 'B': B, 'C': C, 'image_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_list)
