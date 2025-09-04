import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
from datetime import datetime
# Modified: Conditional import based on model
# from model.DDGCRN import DDGCRN as Network  # Removed; now conditional below
from model.DDGCRN_PP import Trainer  # Use the fixed base Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
import warnings
warnings.filterwarnings('ignore')

#*************************************************************************#

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

# Helper function to clean config values (strip spaces and remove inline comments)
def clean_config_value(value):
    # Remove everything after ';' (inline comment) and strip spaces
    if ';' in value:
        value = value.split(';', 1)[0]
    return value.strip()

#parser
parser = argparse.ArgumentParser(description='arguments')  # Renamed to 'parser' for clarity
parser.add_argument('--dataset', default='PEMSD4', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
parser.add_argument('--debug', default='False', type=eval)
parser.add_argument('--model', default='DDGCRN', type=str)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()  # Fixed: Use 'args' consistently (removed args1)

#get configuration
config_file = './config_file/{}_{}.conf'.format(args.dataset, args.model)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

#data (apply cleaning to all config values)
parser.add_argument('--val_ratio', default=float(clean_config_value(config['data']['val_ratio'])), type=float)
parser.add_argument('--test_ratio', default=float(clean_config_value(config['data']['test_ratio'])), type=float)
parser.add_argument('--lag', default=int(clean_config_value(config['data']['lag'])), type=int)
parser.add_argument('--horizon', default=int(clean_config_value(config['data']['horizon'])), type=int)
parser.add_argument('--num_nodes', default=int(clean_config_value(config['data']['num_nodes'])), type=int)
parser.add_argument('--tod', default=eval(clean_config_value(config['data']['tod'])), type=eval)
parser.add_argument('--normalizer', default=clean_config_value(config['data']['normalizer']), type=str)
parser.add_argument('--column_wise', default=eval(clean_config_value(config['data']['column_wise'])), type=eval)
parser.add_argument('--default_graph', default=eval(clean_config_value(config['data']['default_graph'])), type=eval)
parser.add_argument('--steps_per_day', default=int(clean_config_value(config['data']['steps_per_day'])), type=int)
parser.add_argument('--days_per_week', default=int(clean_config_value(config['data']['days_per_week'])), type=int)
#model
parser.add_argument('--input_dim', default=int(clean_config_value(config['model']['input_dim'])), type=int)
parser.add_argument('--output_dim', default=int(clean_config_value(config['model']['output_dim'])), type=int)
parser.add_argument('--embed_dim', default=int(clean_config_value(config['model']['embed_dim'])), type=int)
parser.add_argument('--rnn_units', default=int(clean_config_value(config['model']['rnn_units'])), type=int)
parser.add_argument('--num_layers', default=int(clean_config_value(config['model']['num_layers'])), type=int)
parser.add_argument('--cheb_k', default=int(clean_config_value(config['model']['cheb_order'])), type=int)
parser.add_argument('--use_day', default=eval(clean_config_value(config['model']['use_day'])), type=eval)
parser.add_argument('--use_week', default=eval(clean_config_value(config['model']['use_week'])), type=eval)
# New args for DDGCRN_PP
parser.add_argument('--grid_h', default=int(clean_config_value(config['model'].get('grid_h', '8'))), type=int)  # Default 8 if not in config
parser.add_argument('--grid_w', default=int(clean_config_value(config['model'].get('grid_w', '8'))), type=int)  # Default 8 if not in config
parser.add_argument('--topk', default=int(clean_config_value(config['model'].get('topk', '16'))), type=int)
parser.add_argument('--regions', default=int(clean_config_value(config['model'].get('regions', '4'))), type=int)
#train
parser.add_argument('--loss_func', default=clean_config_value(config['train']['loss_func']), type=str)
parser.add_argument('--seed', default=int(clean_config_value(config['train']['seed'])), type=int)
parser.add_argument('--batch_size', default=int(clean_config_value(config['train']['batch_size'])), type=int)
parser.add_argument('--epochs', default=int(clean_config_value(config['train']['epochs'])), type=int)
parser.add_argument('--lr_init', default=float(clean_config_value(config['train']['lr_init'])), type=float)
parser.add_argument('--weight_decay', default=float(clean_config_value(config['train']['weight_decay'])), type=float)
parser.add_argument('--lr_decay', default=eval(clean_config_value(config['train']['lr_decay'])), type=eval)
parser.add_argument('--lr_decay_rate', default=float(clean_config_value(config['train']['lr_decay_rate'])), type=float)
parser.add_argument('--lr_decay_step', default=clean_config_value(config['train']['lr_decay_step']), type=str)
parser.add_argument('--early_stop', default=eval(clean_config_value(config['train']['early_stop'])), type=eval)
parser.add_argument('--early_stop_patience', default=int(clean_config_value(config['train']['early_stop_patience'])), type=int)
parser.add_argument('--grad_norm', default=eval(clean_config_value(config['train']['grad_norm'])), type=eval)
parser.add_argument('--max_grad_norm', default=int(clean_config_value(config['train']['max_grad_norm'])), type=int)
parser.add_argument('--teacher_forcing', default=False, type=bool)
parser.add_argument('--real_value', default=eval(clean_config_value(config['train']['real_value'])), type=eval, help = 'use real value for loss calculation')
#test
parser.add_argument('--mae_thresh', default=eval(clean_config_value(config['test']['mae_thresh'])), type=eval)
parser.add_argument('--mape_thresh', default=float(clean_config_value(config['test']['mape_thresh'])), type=float)
#log
parser.add_argument('--log_dir', default='./', type=str)
parser.add_argument('--log_step', default=int(clean_config_value(config['log']['log_step'])), type=int)
parser.add_argument('--plot', default=eval(clean_config_value(config['log']['plot'])), type=eval)
args = parser.parse_args()  # Parse again after adding all args

print(args)  # Debug: Print parsed args

# init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5:]))
else:
    args.device = 'cpu'

# New: Conditional model import
if args.model == 'DDGCRN':
    from model.DDGCRN import DDGCRN as Network
elif args.model == 'DDGCRN_PP':
    from model.DDGCRN_PP import DDGCRN_PP as Network
else:
    raise ValueError(f"Unknown model: {args.model}")

#init model
model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
        # Modified: Use --model_path if provided, else default to the specific trained model path
    load_path = '/media/external_16TB_1/zafarany/DDGCRN/experiments/PEMSD4/20250901143839best_model.pth'
    print(f"Loading saved model from {load_path}")
    trainer.logger.info(f"Loading saved model from {load_path}")
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
