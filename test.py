from config_tode_test import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utility
from model import get as get_model
from data import get as get_data
from loss import get as get_loss
from summary import get as get_summary
from metric import get as get_metric
import matplotlib.pyplot as plt
# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from torch.cuda.amp import autocast
import argparse
from opt import *
import data.cleargrasp_synthetic_dataset as cleargrasp_syn
import data.mixed_dataset as MixedDataset
from datasets.tode_dataset import get_dataset
from model.backbone.tode_zl import tode_zl
# Minimize randomness
torch.manual_seed(args_config.seed)
np.random.seed(args_config.seed)
random.seed(args_config.seed)
torch.cuda.manual_seed_all(args_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
GLOBAL_TRAIN_STEP = 0

def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            if args.force_maxdepth:
                new_args.max_depth = args.max_depth
                
            new_args.start_epoch = args.start_epoch
    return new_args



import logging
import yaml
from datasets.logger import ColoredLogger
from tode_utils.metrics import MetricsRecorder

def get_metrics(metrics_params = None):
    """
    Get the metrics settings from configuration.

    Parameters
    ----------

    metrics_params: dict, optional, default: None. If metrics_params is provided, then use the parameters specified in the metrics_params to get the metrics. Otherwise, the metrics parameters in the self.params will be used to get the metrics.
    
    Returns
    -------

    A MetricsRecorder object.
    """

    metrics_list = metrics_params.get('types', ['MSE', 'MaskedMSE', 'RMSE', 'MaskedRMSE', 'REL', 'MaskedREL', 'MAE', 'MaskedMAE', 'Threshold@1.05', 'MaskedThreshold@1.05', 'Threshold@1.10', 'MaskedThreshold@1.10', 'Threshold@1.25', 'MaskedThreshold@1.25'])
    metrics = MetricsRecorder(metrics_list = metrics_list, **metrics_params)
    return metrics

def test(args):

    
    tode_encoder = tode_zl()
    checkpoint_file = './tode_checkpoints/checkpoint-epoch25.tar'
    print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    tode_encoder.load_state_dict(checkpoint['model_state_dict'])
    tode_encoder = tode_encoder.cuda()
    tode_encoder.eval()
    # Prepare dataset
    cfg_filename = './configs/train_cgsyn+ood_val_cgreal.yaml'
    with open(cfg_filename, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    dataset_params = cfg_params.get('dataset', {'data_dir': 'data'})
    # test_dataset = get_dataset(dataset_params,split='test') # cg real
    test_dataset = get_dataset(dataset_params,split='test') # cg real

    # loader_test = DataLoader(dataset=test_dataset, batch_size=1,
    #                          shuffle=False, num_workers=0)
    loader_test = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=True, num_workers=0)
    metrics_params = cfg_params.get('metrics')
    metrics = get_metrics(metrics_params)
    # Network
    model = get_model(args)
    net = model(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

    net = nn.DataParallel(net)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    metric_names = (['MaskedMSE', 'MaskedRMSE', 'MaskedREL', 'MaskedMAE', 'MaskedThreshold@1.05', 'MaskedThreshold@1.10', 'MaskedThreshold@1.25'])
    writer_test = summary(args.save_dir+'debug', 'test', args, None, metric_names)

    net.eval()
    # net.module.depth_backbone.train()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        t0 = time.time()
        
        if args.opt_level == 'O0':
            with torch.no_grad():
                feat = tode_encoder(sample['rgb'], sample['depth'].unsqueeze(1))
                sample['feat'] = feat
                output = net(sample)
        else:
            with autocast():
                output = net(sample)
        t1 = time.time()
        sample['pred'] = output['pred'].squeeze(1)
        t_total += (t1 - t0)
        depth_gt_mask = sample['depth_gt_mask'][0].detach().cpu()
        depth_gt = sample['depth_gt'][0].detach().cpu()
        pred = sample['pred'][0].detach().cpu()
        zero_mask = depth_gt>0
        mask = depth_gt_mask & zero_mask
        for i in range(len(output['pred'])):
            sample['pred'] = output['pred'][i].squeeze(1)
            pred = sample['pred'][0].detach().cpu()
            rmse = torch.sqrt((depth_gt - pred)**2) * mask
            print(rmse[mask!=0].mean())
        # _ = metrics.evaluate_batch(sample, record = True)
        metric_val = metrics.evaluate_batch(sample, record = True)
        metric_val = torch.tensor(list(metric_val.values())[1:]).cuda().unsqueeze(0)
        writer_test.add(None, metric_val)
        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        pbar.set_description(error_str)
        pbar.update(loader_test.batch_size)
    
    metrics_result = metrics.get_results()
    # writer_test.update(args.epochs, sample, output)
    metrics.display_results()
    pbar.close()

def main(args):
    result_list = test(args)
    
if __name__ == '__main__':


    args_main = check_args(args_config)
    # args_config

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)
