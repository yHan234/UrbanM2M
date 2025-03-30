"""run this file to generate tiles for training and testing
it may take 5 minute to generate tiles from an area of 10000km^2"""
import os, sys
from os.path import join as pj
import pandas as pd, numpy as np
import random as rd
import shutil
from tqdm import tqdm
import json
from gdal_util import *

try:
    from osgeo import gdal
except:
    import gdal

NO_DATA_VALUE = 0


def is_complete(region: np.ndarray, water: np.ndarray):
    if (water == 1).sum() >= 0.8 * water.size:
        return False
    if (region == 1).sum() == 0:
        return False
    return True


def save_blocks(indi_block_dir: str,
                var_name: str,
                arr: np.ndarray,
                start_row: int,
                start_col: int,
                process_func=None
                ) -> None:
    if not os.path.exists(indi_block_dir):
        os.makedirs(indi_block_dir)
    out_path = pj(indi_block_dir, var_name)
    if process_func:
        arr = process_func(arr.copy())
    img_saver.save_block(arr, out_path, start_row=start_row, start_col=start_col)

"""useless backup start"""
flip_funcs = [lambda x: np.fliplr(x), lambda x: np.flipud(x)]
# 90, 180, 270
rotate_funcs = [lambda x: np.rot90(x), lambda x: np.flipud(x)[:, ::-1], lambda x: np.rot90(x)[::-1, ::-1]]

def splits():
    pass

def crops():
    pass
"""end"""

def loop_blocks(rasters: dict,
                block_size: int,
                block_step: int,
                year_range: range = range(2014, 2016),
                spa_vars: list = None,
                region_dir: str = None,
                required_block_count: int = 10000,
                crop:bool = True) -> None:
    if crop:
        assert block_size == block_step, "step != size"
    
    region = rasters['range']
    water = rasters['water']
    if crop:
        root_dir = pj(region_dir, f'blockCrop{block_size}_{block_step}')
    else:
        root_dir = pj(region_dir, f'block{block_size}_{block_step}')
        
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    else:
        shutil.rmtree(root_dir)
        os.mkdir(root_dir)
    
    valid_block_count = 0
    existing_block = {}
    max_start_row, max_start_col = region.shape[0], region.shape[1]
    block_rcs = []
    
    if crop:
        while valid_block_count < required_block_count:
            row_start, col_start = rd.randint(block_size, max_start_row), rd.randint(block_size, max_start_col)
            row_end, col_end = row_start + block_size, col_start + block_size
            if f'{row_end}_{col_end}' in existing_block.keys():
                continue
            else:
                existing_block[f'{row_end}_{col_end}'] = 1
            region_block = region[row_start:row_end, col_start:col_end]
            water_block = water[row_start:row_end, col_start:col_end]
            if is_complete(region_block, water_block):  # save current tile or not
                block_rcs.append([row_end, col_end])
                valid_block_count += 1
            print(row_end, col_end)
    else:
        for i, row_end in enumerate(range(block_step, max_start_row, block_step)):
            for j, col_end in enumerate(range(block_step, max_start_col, block_step)):
                row_start, col_start = row_end - block_size, col_end - block_size
                region_block = region[row_start:row_end, col_start:col_end]
                water_block = water[row_start:row_end, col_start:col_end]
                if is_complete(region_block, water_block):  #  save current tile or not
                    block_rcs.append([row_end, col_end])
                    print(row_end ,col_end)
                
    for row_end, col_end in block_rcs:
        row_start, col_start = row_end - block_size, col_end - block_size
        flip_rdint, rotate_rdint = rd.randint(0, 1), rd.randint(0, 2)
        process_list = [[pj(root_dir, f'{row_end}_{col_end}'), None],
                            [pj(root_dir, f'{row_end}_{col_end}_f{flip_rdint}'), flip_funcs[flip_rdint]],
                            [pj(root_dir, f'{row_end}_{col_end}_r{flip_rdint}'), rotate_funcs[flip_rdint]]]
        sample_type = rd.randint(1, 3) if crop else 1
        for var in spa_vars:  #
            var_block = rasters[var][row_start:row_end, col_start:col_end]
            for i in range(sample_type):
                save_blocks(process_list[i][0], f'{var}.tif', var_block, row_start, col_start, process_list[i][1])
        ts_info = {'year_start': year_range.start, 'year_end': year_range.stop - 1, 'urban_props': [],
                   'expand_rate': []}
        for year in year_range:  #
            land_block = rasters[year][row_start:row_end, col_start:col_end]
            urban_prop = int((land_block == 1).sum()) / (block_size ** 2)
            ts_info['urban_props'].append(urban_prop)
            ts_info['expand_rate'].append(urban_prop - ts_info['urban_props'][0])
            for i in range(sample_type):
                save_blocks(process_list[i][0], f'land_{year}.tif', land_block, row_start, col_start,
                            process_list[i][1])
        for i in range(sample_type):
            info_path = pj(process_list[i][0], 'info.json')
            with open(info_path, 'w') as f:
                json.dump(ts_info, f)
        print(f'{row_end},{col_end} ok')



def read_rasters(raster_paths: dict) -> dict:

    raster_arrays = {}
    for ras_name, ras in raster_paths.items():
        arr = GDALImage.read_single(ras, NO_DATA_VALUE)
        raster_arrays[ras_name] = arr
        print(ras_name)
    return raster_arrays


# def read_rasters2(year_range: range, region_dir: str, spa_vars: list):
#     year_arrs = {}
#     for year in year_range:
#         land_path = pj(region_dir, 'year', f'land_{year}.tif')
#         if not os.path.exists(land_path):
#             raise RuntimeError('failed to find urban extent map')
#         year_arrs[year] = GDALImage.read_single(var_path, NO_DATA_VALUE)
#
#
#     var_arrs = {}
#     for var in spa_vars:
#         var_path = pj(region_dir, rf'{var}.tif')
#         if not os.path.exists(var_path):
#             raise RuntimeError('failed to find spatial variable')
#         var_arrs[var] = GDALImage.read_single(var_path, NO_DATA_VALUE)



def get_ras_paths(year_range: range, region_dir: str, spa_vars: list) -> dict:
    res = {}
    for var in spa_vars:
        res[var] = pj(region_dir, rf'{var}.tif')
    # 
    res['range'] = pj(region_dir, 'range.tif')
    res['water'] = pj(region_dir, 'water.tif')
    # 
    for year in year_range:
        land_year_name = f'{year}_hzb.tif'
        land_ras_name = pj(region_dir, 'year', land_year_name)
        res[year] = land_ras_name
    return res


# setting
spa_vars = ['Dcity', 'DCounty', 'Dhigh', 'DPrimary', 'DSecondary', 'DTertiary', 'dem', 'gdp', 'pop', 'slope', 'water']
region_dir = '/root/autodl-tmp/hzb'
img_saver = GDALImage(pj(region_dir, f'{spa_vars[0]}.tif'))
# split training set
year_range = range(2000, 2012)
raster_paths = get_ras_paths(year_range, region_dir, spa_vars)
rasters = read_rasters(raster_paths)
loop_blocks(rasters, 64, 64, year_range, spa_vars, region_dir, crop = False)

# split testing set
year_range = range(2006, 2018)
raster_paths = get_ras_paths(year_range, region_dir, spa_vars)
rasters = read_rasters(raster_paths)
loop_blocks(rasters, 64, 38, year_range, spa_vars, region_dir, crop = False)

