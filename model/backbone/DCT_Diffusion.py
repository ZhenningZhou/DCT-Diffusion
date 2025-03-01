# Copyright (c) Phigent Robotics. All rights reserved.
# for debug
# import sys
# sys.path.append('/mnt/cfs/algorithm/yiqun.duan/depth/diffuserdc/src')

import torch
from torch import nn
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import torch.utils.checkpoint as checkpoint
from mmdet.models import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from model.ops.cbam import CBAMWithPosEmbed
from model.backbone.swin_transformer import SwinTransformer
import math
from diffusers.models.embeddings import TimestepEmbedding, Timesteps



class UpSampleBN(nn.Module):
    def __init__(self, input_features, output_features, res = True):
        super(UpSampleBN, self).__init__()
        self.res = res

        self._net = nn.Sequential(nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                #   nn.BatchNorm2d(input_features),
                                  nn.GroupNorm(8,input_features),
                                #   nn.SiLU(),
                                  nn.GELU(),
                                  nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                #   nn.BatchNorm2d(input_features),
                                  nn.GroupNorm(8,input_features),
                                  nn.GELU(),
                                #   nn.SiLU()
                                  )

        self.up_net = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),
                                    # nn.BatchNorm2d(output_features, output_features),
                                    nn.GroupNorm(8, output_features),
                                    nn.GELU()
                                    # nn.SiLU(True)
                                    )


    def forward(self, x, concat_with):
        if concat_with == None:
            if self.res:
                conv_x = self._net(x) + x
            else:
                conv_x = self._net(x)
        else:
            if self.res:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) + torch.cat([x, concat_with], dim=1)
            else:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) 

        return self.up_net(conv_x)

class SELayer_down(nn.Module):
    def __init__(self, H, W):
        super(SELayer_down, self).__init__()
        self.avg_pool_channel = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((H//2, W//2))

    def forward(self, in_data, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view(b, -1, c)
        y = self.avg_pool_channel(x).view(b, h, w, 1)
        y = y.permute(0, 3, 1, 2)
        y = self.avg_pool_2d(y)
        return in_data * y.expand_as(in_data)





class DecoderBN(nn.Module):
    def __init__(self, num_features=128, lambda_val=1, res=True):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.lambda_val = lambda_val

        self.se1_down = SELayer_down(120, 160)
        self.se2_down = SELayer_down(60, 80)
        self.se3_down = SELayer_down(30, 40)

        self.up1 = UpSampleBN(192, features, res)
        self.up2 = UpSampleBN(features + 96, features, res)
        self.up3 = UpSampleBN(features + 48, features, res)
        self.up4 = UpSampleBN(features + 24, features//2, res)


    def forward(self, features):
        x_block4, x_block3, x_block2, x_block1= features[3], features[2], features[1], features[0]

        x_block2_1 = self.lambda_val * self.se1_down(x_block2, x_block1) + (1-self.lambda_val) * x_block2
        x_block3_1 = self.lambda_val * self.se2_down(x_block3, x_block2_1) + (1-self.lambda_val) * x_block3
        x_block4_1 = self.lambda_val * self.se3_down(x_block4, x_block3_1) + (1-self.lambda_val) * x_block4
        # x_block4_1 bottleneck feature
        x_d0 = self.up1(x_block4_1, None)
        x_d1 = self.up2(x_d0, x_block3_1)
        x_d2 = self.up3(x_d1, x_block2_1)
        x_d3 = self.up4(x_d2, x_block1)


        return x_d3

@BACKBONES.register_module()
class Tode_backbone(nn.Module):
    def __init__(self, lambda_val = 1, res = True):
        super(Tode_backbone, self).__init__()

        self.encoder = SwinTransformer(patch_size=2, in_chans= 1, embed_dim=24)
        self.decoder = DecoderBN(num_features=128, lambda_val=lambda_val, res=res)
        self.linear0 = nn.Linear(256,24)
        self.linear1 = nn.Linear(256,48)
        self.linear2 = nn.Linear(256,96)
        self.linear3 = nn.Linear(256,192)
        self.linearlist = [self.linear0, self.linear1, self.linear2, self.linear3]
        self.nonlinear = nn.GELU()
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            # nn.BatchNorm2d(64, 64),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            # nn.ReLU(True),
            # nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size = 3, stride = 1, padding = 1)
            # nn.SiLU(True)
        )
        self.time_proj = Timesteps(128, flip_sin_to_cos =True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(
            128,
            256,
            act_fn='gelu',
            post_act_fn=None,
            cond_proj_dim=None,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, noisy_img, t, feat):
        if len(t.shape) ==0:
            t = t.unsqueeze(0)
        t_emb = self.time_proj(t)
        t_emb = self.time_embedding(t_emb)
        if len(noisy_img.shape) == 3:
            n, h, w = noisy_img.shape
            noisy_img = noisy_img.view(n, 1, h, w)
        noisy_img = noisy_img
        encoder_x = self.encoder(noisy_img)



        
        
        assert len(encoder_x) == len(feat)
        for i in range(len(encoder_x)):
   
            encoder_x[i] += (feat[i] +self.linearlist[i](self.nonlinear(t_emb))[...,None,None])
        decoder_x = self.decoder(encoder_x)

        out = self.final(decoder_x)


        # return encoder_x
        return out


    @classmethod
    def build(cls, **kwargs):
 
        print('Building Encoder-Decoder model..', end='')
        m = cls(**kwargs)
        print('Done.')
        return m



def tode_backbone():
    net = Tode_backbone()
    return net

if __name__ == '__main__':
    pass