import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """ 正余弦时间步编码 """
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (batch_size, 1) 时间步输入
        返回: (batch_size, dim) 编码后向量
        """
        half_dim = self.dim // 2
        exp_term = torch.exp(-math.log(10000) * torch.arange(half_dim).float() / half_dim).to(t.device)
        sin_embedding = torch.sin(t * exp_term)
        cos_embedding = torch.cos(t * exp_term)
        pos_encoding = torch.cat([sin_embedding, cos_embedding], dim=-1)
        return pos_encoding.unsqueeze(-1).unsqueeze(-1)  # (batch, dim, 1, 1)


class UncertainAwareAttention(nn.Module):
    """ Uncertain-aware Attention """
    def __init__(self, in_channels):
        super(UncertainAwareAttention, self).__init__()
        self.conv1x1_a = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_b = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, 32, kernel_size=3, padding=1)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        branch1 = self.conv1x1_a(x)
        branch2 = self.conv1x1_b(x)
        concat = torch.cat([branch1, branch2], dim=1)
        conv3x3 = self.conv3x3(concat)
        out = self.bn_relu(conv3x3)
        out = self.conv1x1_out(out)
        attention = self.sigmoid(out)
        return attention * x


class TimestepAwareAttention(nn.Module):
    def __init__(self, in_channels, pos_enc_dim=32):
        super(TimestepAwareAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(pos_enc_dim)
        self.conv3x3 = nn.Conv2d(in_channels + pos_enc_dim, 32, kernel_size=3, padding=1)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        # t 是 (batch, 1)，进行正余弦编码
        t_emb = self.positional_encoding(t).expand(-1, -1, x.shape[2], x.shape[3])  # (batch, 32, H/2, W/2)

        concat = torch.cat([x, t_emb], dim=1)
        conv3x3 = self.conv3x3(concat)
        out = self.bn_relu(conv3x3)
        out = self.conv1x1_out(out)
        attention = self.sigmoid(out)
        return attention * x


class ConditionedFusionModule(nn.Module):
    def __init__(self, in_channels_xt, in_channels_xvis):
        super(ConditionedFusionModule, self).__init__()
        self.uncertain_attention = UncertainAwareAttention(in_channels_xt)
        self.timestep_attention = TimestepAwareAttention(in_channels_xt)  # 注意这里改为 in_channels_xt

        self.conv1x1_xvis = nn.Conv2d(in_channels_xvis, in_channels_xt, kernel_size=1)
        self.bn_relu_xvis = nn.Sequential(nn.BatchNorm2d(in_channels_xt), nn.ReLU(inplace=True))

        self.conv3x3_fusion1 = nn.Conv2d(in_channels_xt, in_channels_xt, kernel_size=3, padding=1)
        self.bn_relu_fusion1 = nn.Sequential(nn.BatchNorm2d(in_channels_xt), nn.ReLU(inplace=True))

        self.conv3x3_fusion2 = nn.Conv2d(in_channels_xt, in_channels_xt, kernel_size=3, padding=1)
        self.bn_relu_fusion2 = nn.Sequential(nn.BatchNorm2d(in_channels_xt), nn.ReLU(inplace=True))

    def forward(self, xt, xvis, t):
        xt_atten = self.uncertain_attention(xt)

        xvis = self.conv1x1_xvis(xvis)
        xvis = self.bn_relu_xvis(xvis)
        xvis_atten = self.timestep_attention(xvis, t)

        xt_fused = xt_atten + xvis_atten  # 替换 element-wise 乘法为加法

        out = self.conv3x3_fusion1(xt_fused)
        out = self.bn_relu_fusion1(out)
        out = self.conv3x3_fusion2(out)
        out = self.bn_relu_fusion2(out)

        return out

# # test
# if __name__ == "__main__":
#     batch_size, c_xt, h, w = 1, 16, 64, 64  # Xt 通道为 16
#     c_xvis = 256  # Xvis 通道为 256
#     xt = torch.randn(batch_size, c_xt, h, w)
#     xvis = torch.randn(batch_size, c_xvis, h, w)
#     t = torch.randn(batch_size, 1)  # 单个时间步输入
#
#     model = ConditionedFusionModule(in_channels_xt=16, in_channels_xvis=256)
#     output = model(xt, xvis, t)
#     print("Output shape:", output.shape)  # 应该是 (1, 16, 64, 64)