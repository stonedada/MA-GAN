"""This module contains simple helper functions """
from __future__ import print_function
import pandas as pd
import logging
from util.custom_metrics import *
from skimage import metrics
import torch
import numpy as np
from PIL import Image
import os
import tifffile

DF_NAMES = ["NRMSE",
            "SSIM",
            "PCC",
            "Dice",
            "PSNR",
            "r2"
            ]


def tensor2im(input_image, imtype=np.uint16):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = np.rint((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 32767.0)  # post-processing: transpose and scaling
        # image_numpy = (image_numpy - image_numpy.min())
        image_numpy = np.rint((image_numpy + 1.0) / 2.0 * 255.0)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2numpy(input_image, imtype=np.uint16):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)
    tifffile.imsave(image_path, image_numpy)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_dataframe(nbr_rows=None, df_names=DF_NAMES):
    """
    Create empty frames metadata pandas dataframe given number of rows
    and standard column names defined below

    :param [None, int] nbr_rows: The number of rows in the dataframe
    :param list df_names: Dataframe column names
    :return dataframe frames_meta: Empty dataframe with given
        indices and column names
    """

    if nbr_rows is not None:
        # Create empty dataframe
        frames_meta = pd.DataFrame(
            index=range(nbr_rows),
            columns=df_names,
        )
    else:
        frames_meta = pd.DataFrame(columns=df_names)
    return frames_meta


def load(output_folder, cuda):
    """
    Loads a previous network model from the given folder. This folder should contain : params.net.

    :param output_folder: The path of the folder containing the network
    :param cuda: Wheter to use CUDA

    :returns : The parameters of the network
    """
    net_params = torch.load(os.path.join(output_folder, "params.net"),
                            map_location=None if cuda else "cpu")
    return net_params


def Metrics(iter_num, frames_meta, visuals, save_path):
    row = {}
    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        row[label] = image_numpy
    # real_A = row['real_A'].squeeze()
    # fake_B = row['fake_B'].squeeze()
    # real_B = row['real_B'].squeeze()
    real_A = row['confocal'].squeeze()
    fake_B = row['fakeSTED'].squeeze()
    real_B = row['STED'].squeeze()

    # evaluate
    dice = dice_metric(real_B, fake_B)
    psnr = metrics.peak_signal_noise_ratio(real_B, fake_B, data_range=real_B.max() - real_B.min())
    nrmse = metrics.normalized_root_mse(real_B, fake_B)
    ssim = metrics.structural_similarity(real_B, fake_B, win_size=21,
                                         data_range=real_B.max() - real_B.min())
    pcc = pearsonr(real_B.flatten(), fake_B.flatten())
    r2 = r2_metric(real_B, fake_B)

    # Save tif file
    tifffile.imwrite(f'{save_path}/{iter_num}_fake_B.tif', data=fake_B)
    tifffile.imwrite(f'{save_path}/{iter_num}_real_A.tif', data=real_A)
    tifffile.imwrite(f'{save_path}/{iter_num}_real_B.tif', data=real_B)

    # Save csv.file
    meta_row = dict.fromkeys(DF_NAMES)
    meta_row['NRMSE'] = nrmse
    meta_row['SSIM'] = ssim
    meta_row['PCC'] = pcc
    meta_row['Dice'] = dice
    meta_row['PSNR'] = psnr
    meta_row['r2'] = r2
    frames_meta.loc[iter_num] = meta_row

    logging.info('iteration %d : NRMSE: %f,ssim: %f,PCC: %f,dice: %f,PSNR: %f,R2: %f' % (
        iter_num, nrmse, ssim, pcc, dice, psnr, r2))
