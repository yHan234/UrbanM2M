import os, sys
from config import *
from model.CNN_LSTM import CNN_LSTM
from utils.data_loading import TrainDataset
from utils.trainers.trainer import Trainer
import argparse
from datetime import datetime


def check_args(args):
    args.input_tifs   = [f'land_{year}.tif' for year in range(args.start_year,
                                                              args.start_year + args.enc_len + args.fore_len, 1)]
    args.spa_var_tifs = [item + '.tif' for item in args.spa_vars.split('|')]
    args.band = 1 + len(args.spa_var_tifs)
    formatted_date = datetime.now().strftime("%m_%d_%H")
    args.model_name = args.model_type + f'-fs{args.filter_size}-t{formatted_date}'
    args.model_dir  = os.path.join('/root/autodl-tmp/trained_models', args.model_type)
    args.use_att = eval(args.use_att)
    args.use_ce = eval(args.use_ce)
    args.use_mix = True
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input info
    parser.add_argument('--start_year', default=2000, type=int, help='start year of training')
    parser.add_argument('--enc_len',    default=6,    type=int, help='input years')
    parser.add_argument('--fore_len',   default=6,    type=int, help='output years')
    # img info
    parser.add_argument('--height',     default=64,   type=int, help='raster tile size')
    parser.add_argument('--spa_vars',   default='Dcity|DCounty|Dhigh|DPrimary|DSecondary|DTertiary|dem|gdp|pop|slope|water', type=str, help='spatial variables, split by "|"')
    parser.add_argument('--block_dir',  default=r'/root/autodl-tmp/hzb/block64_64', type=str, help='raster tile dir root')
    # model struct
    parser.add_argument('--nlayers',    default=2,     type=int, help='number of layers')
    parser.add_argument('--filter_size',default=5,     type=int, help='filter size')
    parser.add_argument('--use_ce',     default='True',type=str, help='')
    parser.add_argument('--use_att',    default='True',type=str, help='')
    # training
    parser.add_argument('--epochs',     default=60,    type=int, help='epochs')
    parser.add_argument('--batch_size', default=8,     type=int, help='batch size')
    parser.add_argument('--lr',         default=0.00005, type=float, help='learning rate')
    parser.add_argument('--eta_decay',  default=0.015,  type=float, help='teacher forcing prob increase through epochs')
    # sampling info
    parser.add_argument('--sample_count',default=5000,  type=int, help='training samples from the total blocks')
    parser.add_argument('--val_prop',   default=0.25,   type=float, help='proportion of validation samples from the selected samples')
    # saving info
    parser.add_argument('--model_type', default='hzb',  type=str, help='model save dir')

    args = parser.parse_args()
    args = check_args(args)
    print(args.spa_vars)

    model = CNN_LSTM(enc_len         = args.enc_len,
                     fore_len        = args.fore_len,
                     s_channels      = len(args.spa_var_tifs),
                     img_height      = args.height,
                     img_width       = args.height,
                     hidden_dim      = args.height)
    model = model.cuda()
    dataset = TrainDataset(args.block_dir,
                        args.input_tifs,
                        args.spa_var_tifs,
                        args.sample_count,
                        args.height)
    t = Trainer(model, args, dataset, device)
    t.loop()