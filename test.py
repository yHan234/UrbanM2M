from config import *

import argparse
import os, sys
from collections import OrderedDict


from utils.data_loading import TrainDataset
from utils.future_land import *
from utils.trainers.tester import *
from model.PredRNN import PredRNN

def check_args(args):

    args.block_dir = os.path.join(args.data_root_dir, f'block{args.height}_{args.block_step}')
    args.gt_dir    = os.path.join(args.data_root_dir, 'year')
    args.gn_dir    = os.path.join(args.data_root_dir, 'sim')
    args.prob_dir  = os.path.join(args.data_root_dir, 'prob')
    args.water_tif = os.path.join(args.data_root_dir, 'water.tif')
    args.range_tif = os.path.join(args.data_root_dir, 'range.tif')

    args.input_tifs = [f'land_{year}.tif' for year in range(args.start_year,
                                          args.start_year + args.enc_len + args.fore_len, 1)]
    args.spa_var_tifs = [item + '.tif' for item in args.spa_vars.split('|')]
    args.band = 1 + len(args.spa_var_tifs)
    args.gt_tifs    = [os.path.join(args.gt_dir, tif)  for tif in args.input_tifs[args.enc_len - 1:]]
    args.gn_tifs    = [os.path.join(args.gn_dir,   f'{args.region}-{args.model_type}', f'sim-' + tif) for tif in args.input_tifs[1:]]
    args.prob_tifs  = [os.path.join(args.prob_dir, f'{args.region}-{args.model_type}', f'prob-' + tif) for tif in args.input_tifs[1:]]
    args.use_mix = True
    args.run_model = bool(eval(args.run_model))
    args.use_att = eval(args.use_att)
    args.use_ce = eval(args.use_ce)
    args.band = 1 + len(args.spa_var_tifs)
    return args

parser = argparse.ArgumentParser()

# input information
parser.add_argument('--start_year', default=2006, type=int)
parser.add_argument('--enc_len',    default=6,    type=int)
parser.add_argument('--fore_len',   default=6,    type=int)
# image information
parser.add_argument('--height',     default=64,   type=int)
parser.add_argument('--block_step', default=38,   type=int)
parser.add_argument('--edge_width', default=4,   type=int)
# data information
parser.add_argument('--spa_vars',   default='Dcity|DCounty|Dhigh|DPrimary|DSecondary|DTertiary|dem|gdp|pop|slope|water', type=str)
parser.add_argument('--region',     default='hzb',          type=str)
parser.add_argument('--model_type', default='hzb',       type=str)
parser.add_argument('--data_root_dir', default='/root/autodl-tmp/hzb',       type=str)
# model struct
parser.add_argument('--nlayers',    default=2,     type=int, help='number of layers')
parser.add_argument('--filter_size',default=5,     type=int, help='filter size')
parser.add_argument('--use_ce',     default='True',type=str, help='')
parser.add_argument('--use_att',    default='True',type=str, help='')

parser.add_argument('--log_file', default='/root/autodl-tmp/test/gba.csv', type=str)
parser.add_argument('--model_path', default='/root/autodl-tmp/trained_models/hzb/hzb-fs5-t04_05_11-e10.pth', type=str) # model dir is required,
# training parameters

parser.add_argument('--run_model',  default='True',  type=str)
parser.add_argument('--numworkers',  default=0,  type=int)
parser.add_argument('--batch_size', default=100,     type=int)

args = parser.parse_args()
args = check_args(args)
# 推理或读取已有图像
saver = GDALImage(args.gt_tifs[0])
gt_arrs = [GDALImage.read_single(tif, 0) for tif in args.gt_tifs]
water_arr = saver.read_single(args.water_tif, 0)
range_arr = saver.read_single(args.range_tif, 0)

def migrate_model(model_weights):
    new_weights = OrderedDict()
    for k, v in model_weights.items():
        # new_weights[k[7:]] = v
        new_weights[k] = v
    return new_weights


if args.run_model:
    # model = torch.load(args.model_path)
    # # model.use_attention = args.use_att
    # model.fore_len = args.fore_len
    # model.cuda()
    
    model_weights = torch.load(args.model_path, map_location='cpu').state_dict()
    new_weights = migrate_model(model_weights)
    model = PredRNN(
        in_channels=args.band,
        num_layers=args.nlayers,
        hidden_size=args.height,
        filter_size=args.filter_size,
        img_width=args.height,
        device=device,
        total_length=args.enc_len + args.fore_len,
        input_length=args.enc_len,
    )
    model.load_state_dict(new_weights)
    model.cuda()
    
    dataset = TrainDataset(args.block_dir,
                        args.input_tifs,
                        args.spa_var_tifs,
                        1,
                        args.height)
    tester = Tester(model, args, dataset, range_arr, device, True)
    tester.loop()
    prob_arrs = tester.prob_arr[args.enc_len - 1:]
    for arr, tif_path in zip(prob_arrs, args.prob_tifs[args.enc_len - 1:]):
        arr[range_arr!=1] = 0
        saver.save_block(arr, tif_path, gdal.GDT_Float32, no_data_value = 0)
else:
    prob_arrs = [saver.read_single(tif_path, 0) for tif_path in args.prob_tifs[args.enc_len - 1:]]

import time
st = time.time()
sim_info  = generate_simulation(prob_arrs,
                    gt_arrs,
                    water_arr,
                    range_arr,
                    saver,
                    args.gn_tifs[args.enc_len - 1:])
print("simulate time:", (time.time()-st)/60, "min")
write_sim_info(args, sim_info)