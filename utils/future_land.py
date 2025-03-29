import numpy as np
from .utils import *
from datetime import datetime
# import pylandstats as pls
import pandas as pd
from copy import deepcopy
import warnings

def calc_land_demand(cur: np.ndarray, prev: np.ndarray) -> int:
    return ((cur - prev) == 1).sum()


def calc_fom(gt: np.ndarray, gn: np.ndarray, prev: np.ndarray):
    diff_count = ((gt - prev) == 1).sum()
    gn_diff = gn - prev
    gt_diff = gt - prev
    rc_sc = ((gn_diff * gt_diff) == 1).sum()
    fom = rc_sc/(2 * (diff_count - rc_sc) + rc_sc)


    return fom

# def cal_landscape(img: np.ndarray,
#                   landscape_metrics:list = ['number_of_patches',
#                                             'perimeter_area_ratio_mn',
#                                             'largest_patch_index'],
#                   landscape_keys:list = ['np', 'paramn', 'lpi']) -> dict:
#     img[img==-9999] = np.nan
#     img_ls = pls.Landscape(img, (30, 30))
#     ls_df  = img_ls.compute_landscape_metrics_df(landscape_metrics)
#     ls_df0 = ls_df.to_dict('index')[0]
#     ls_df = {}
#     for lk, (k, v) in zip(landscape_keys, ls_df0.items()):
#         ls_df[lk] = v
#
#     return ls_df




def get_simulation_result(prev: np.ndarray, prob_map: np.ndarray, water: np.ndarray, land_demand: int) -> np.ndarray:
    sim = deepcopy(prev)
    prob_map[(prev == 1) | (water == 1)] = -9999
    indices = np.unravel_index(np.argsort(-prob_map, axis=None), prob_map.shape)
    indices = np.column_stack(indices)[:land_demand]

    sim[indices[:, 0], indices[:, 1]] = 1
    return sim

def set_random_arr(template:np.ndarray) -> np.ndarray:

    tshape = template.shape
    rd_arr = np.random.rand(tshape[0], tshape[1])
    e_arr  = -np.log(rd_arr)
    rd_arr  = np.power(e_arr, 0.5)

    rec_arr = template * rd_arr
    return rec_arr

def generate_simulation(prob_maps: list,
                        gt_imgs: list,
                        water_map: np.ndarray,
                        range_map: np.ndarray,
                        img_saver: GDALImage,
                        out_tifs: list) -> list:
    prev = gt_imgs[0] # 2010 gt
    st = gt_imgs[0]   # 2010 gt
    prev[range_map != 1] = -9999
    sim_info = []
    tot_land_demand = 0
    prev_gt = gt_imgs[0]
    for prob_map, cur_gt, out_tif in zip(prob_maps, gt_imgs[1:], out_tifs):
        year = out_tif.split('.')[-2].split('_')[-1]

        prob_map = set_random_arr(prob_map)
        cur_gt[range_map != 1] = -9999
        prob_map[range_map != 1] = -9999
        land_demand = calc_land_demand(cur_gt, prev_gt)
        tot_land_demand += land_demand
        cur_sim = get_simulation_result(prev, prob_map, water_map, land_demand)


        img_saver.save_block(cur_sim, out_tif, gdal.GDT_Int16, no_data_value=-9999)
        print('save', out_tif)
        fom = calc_fom(cur_gt, cur_sim, st)
        # metrics = cal_landscape(deepcopy(cur_sim))
        # gt_metrics = cal_landscape(deepcopy(cur_gt), landscape_keys = ['np_ob', 'paramn_ob', 'lpi_ob'])
        print(land_demand, fom)
        sim_info.append({
            'year':year,
            'FoM':fom,
            'total_land_demand':tot_land_demand,
            'land_demand':land_demand,
            'out_path':out_tif,
            })
        # sim_info[-1].update(metrics)
        # sim_info[-1].update(gt_metrics)
        prev = cur_sim
        prev_gt = cur_gt
    return sim_info

def write_sim_info(args, sim_info: list) -> None:
    for i in range(len(sim_info)):
        sim_info[i]['region'] = args.region
        sim_info[i]['model_type'] = args.model_type
    df = pd.DataFrame(sim_info)
    if os.path.exists(args.log_file):
        df0 = pd.read_csv(args.log_file, index_col = False)
        df = pd.concat([df0, df], axis=0)
    else:
        log_dir = os.path.split(args.log_file)[0]
        os.makedirs(log_dir)
        df.to_csv(args.log_file, index=False, header=True)
