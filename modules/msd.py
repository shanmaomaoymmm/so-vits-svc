import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


class ScaleDiscriminator(nn.Module):
    """
    单尺度判别器 - 对下采样后的音频做 1D 卷积判别
    结构与 DiscriminatorS 类似，但使用 avg_pool 下采样
    """
    def __init__(self, norm_f=weight_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器 (MSD) - HiFi-GAN 标准组件
    
    对原始音频做不同倍率的平均池化下采样，
    每个尺度用独立判别器评估，捕捉不同时间尺度的结构特征。
    
    这是解决电子杂音的关键组件 - MPD 关注周期性，
    MSD 关注多尺度波形纹理。
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),   # 原始尺度
            ScaleDiscriminator(),   # 2x 下采样
            ScaleDiscriminator(),   # 4x 下采样
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
