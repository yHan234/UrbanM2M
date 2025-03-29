import numpy as np
import torch


# from config import *
from torch.utils.data import Dataset
import os, re, random, glob
from osgeo import gdal
from os.path import join as pj

def open_single_tif(path: str) -> np.array:
    '''
    '''
    dataset = gdal.Open(path)
    band_count = dataset.RasterCount
    bands = []
    for i in range(1, band_count + 1):
        band = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
        bands.append(band)
    if len(bands) > 1:
        bands = np.stack(bands)
    else:
        bands = bands[0]
    return bands

class TrainDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 inputs: list,
                 spa_vars: list,
                 sample_count: int,
                #  random_proportion: float = 0.5,
                 height: int = 64):
        self.data_dir = data_dir
        self.inputs  = inputs

        self.spa_vars = spa_vars
        self.raw_dirs = os.listdir(data_dir)
        self.height = height
        self.unique_blocks = self.get_unique_blocks()


        # assert 0 < random_proportion <= 1, "incorrect random proportion"
        if sample_count>10:
            self.sampled_blocks = random.sample(self.unique_blocks, sample_count)
        else:
            self.sampled_blocks = self.unique_blocks
        # pattern = '|'.join(self.sampled_blocks)
        # self.data = [os.path.join(data_dir, item) for item in self.raw_dirs if re.search(pattern, item)]
        # pattern = re.compile(r'^([^_]+)_')


        self.data = [os.path.join(data_dir, item) for item in self.sampled_blocks]
        print('loading dataset')

    def get_unique_blocks(self):
        blocks = ["_".join(ddir.split('_')[:2]) for ddir in self.raw_dirs]
        unique_blocks = list(set(blocks))
        return unique_blocks

    def __getitem__(self, idx):
        # return None
        input_tensor   = torch.empty(len(self.inputs), 1, self.height, self.height).float()
        block = self.data[idx]
        rc = os.path.basename(block)

        # spatial variables
        if self.spa_vars:
            spa_var_tensor = torch.empty(len(self.spa_vars), self.height, self.height)
            for i, variable in enumerate(self.spa_vars):
                var_array = open_single_tif(pj(block, variable))
                var_tensor = torch.as_tensor(var_array).float()
                spa_var_tensor[i, :, :] = var_tensor
        else:
            spa_var_tensor = []
        # read by year
        for i, year in enumerate(self.inputs):
            land_arr = open_single_tif(pj(block, year))
            land_tensor = torch.as_tensor(land_arr).float()
            input_tensor[i, 0, :, :]  = land_tensor

        return rc, spa_var_tensor, input_tensor

    def __len__(self):
        return len(self.data)

