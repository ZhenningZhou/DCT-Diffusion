from . import BaseLoss
import torch


class Diffusion_SeparateTODEbase_Loss(BaseLoss):
    def __init__(self, args):
        super(Diffusion_SeparateTODEbase_Loss, self).__init__(args)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            pred = output['pred']
            # gt = sample['gt']
            gt = sample['depth_gt']
            if len(gt.shape) == 3:
                gt = gt.unsqueeze(1)
            mask = sample['depth_gt_mask'].unsqueeze(1)
            # mask = sample['depth_mask']
            if loss_type in ['L1', 'L2', 'Sig']:
                # loss_tmp = loss_func(pred, gt)
                loss_tmp = loss_func(pred, gt, mask)
            elif loss_type in ['DDIM']: 
                loss_tmp = output['ddim_loss']
            elif loss_type in ['SmoothLoss_TODE','L1Loss_Custom','L2Loss_Custom']:
                loss_tmp = loss_func(sample)
            elif loss_type in ['BIN']: 
                # loss_bindepth = output['bin_losses']['loss_depth']
                # loss_binhamfer = output['bin_losses']['loss_chamfer']
                loss_tmp = 0
                for key, value in output['bin_losses'].items():
                    loss_tmp = loss_tmp + value
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
