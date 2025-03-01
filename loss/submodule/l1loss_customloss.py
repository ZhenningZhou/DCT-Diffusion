"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    L2 loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tode_utils.functions import get_surface_normal_from_depth


class L1Loss_CustomLoss(nn.Module):
    def __init__(self, args):
        super(L1Loss_CustomLoss, self).__init__()

    def _l1(self, pred, gt):
        """
        L1 loss in pixel-wise representations.
        """
        return torch.abs(pred - gt)
    
    def forward(self, data_dict, *args, **kwargs):
        """
        Custom masked L1 loss.
        
        Parameters
        ----------

        data_dict: the data dict for computing L1 loss.

        Returns
        -------

        The custom masked L1 loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        pred = torch.clamp(pred, min=0, max = 10.0)
        gt = torch.clamp(gt, min=0, max = 10.0)
        mask = data_dict['loss_mask']
        zero_mask = data_dict['zero_mask']
        # delta = 1.01
        # thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        # mask = (thres > delta) & mask
        loss = self._l1(pred, gt)
        # loss_flatten = loss[mask].view(-1)
        # k = int(mask.sum() * 0.1)
        # loss_topk, _ = torch.topk(loss_flatten, k)

        return loss[mask].mean() + 0.01 * loss[zero_mask].mean()
    

