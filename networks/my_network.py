import torch
import torch.nn.functional as F
from torch import nn

from .layers import *
from .networks1 import weights_init

#基础的模型结构
import torch
import torch.nn.functional as F
from torch import nn


class SharedEncoder(nn.Module):
    def __init__(self, nin, nG=32, has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout
        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG, nG*2, stride=2),
                                   convBatch(nG*2, nG*2))
        self.conv2 = nn.Sequential(convBatch(nG*2, nG*4, stride=2),
                                   convBatch(nG*4, nG*4))

        self.bridge = nn.Sequential(
            convBatch(nG*4, nG*8, stride=2),
            residualConv(nG*8, nG*8),
            convBatch(nG*8, nG*8)
        )
        self.dropout = nn.Dropout2d(p=0.5) if has_dropout else nn.Identity()

    def forward(self, x):
        x = x.float()
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        feats = self.bridge(x2)
        feats = self.dropout(feats)
        return feats, x0, x1, x2


class SegmentationDecoder(nn.Module):
    def __init__(self, n_classes, nG=32):
        super().__init__()
        self.deconv1 = upSampleConv(nG*8, nG*8)
        self.conv5 = nn.Sequential(convBatch(nG*8+nG*4, nG*4),
                                   convBatch(nG*4, nG*4))
        self.deconv2 = upSampleConv(nG*4, nG*4)
        self.conv6 = nn.Sequential(convBatch(nG*4+nG*2, nG*2),
                                   convBatch(nG*2, nG*2))
        self.deconv3 = upSampleConv(nG*2, nG*2)
        self.conv7 = nn.Sequential(convBatch(nG*2+nG*1, nG*1),
                                   convBatch(nG*1, nG*1))
        self.out_conv = nn.Conv2d(nG, n_classes, kernel_size=1)

    def forward(self, feats, x0, x1, x2):
        y = self.deconv1(feats)
        y = self.deconv2(self.conv5(torch.cat([y, x2], dim=1)))
        y = self.deconv3(self.conv6(torch.cat([y, x1], dim=1)))
        y = self.conv7(torch.cat([y, x0], dim=1))
        return self.out_conv(y)


class RegressionDecoder(nn.Module):
    def __init__(self, n_sdf_channels, n_kp_channels, nG=32):
        super().__init__()
        # Shared upsampling path
        self.deconv1 = upSampleConv(nG*8, nG*8)
        self.conv5 = nn.Sequential(convBatch(nG*8+nG*4, nG*4),
                                   convBatch(nG*4, nG*4))
        self.deconv2 = upSampleConv(nG*4, nG*4)
        self.conv6 = nn.Sequential(convBatch(nG*4+nG*2, nG*2),
                                   convBatch(nG*2, nG*2))
        self.deconv3 = upSampleConv(nG*2, nG*2)
        self.conv7 = nn.Sequential(convBatch(nG*2+nG*1, nG*1),
                                   convBatch(nG*1, nG*1))
        # Heads
        self.sdf_head = nn.Conv2d(nG, n_sdf_channels, kernel_size=1)
        self.kp_head = nn.Conv2d(nG, n_kp_channels, kernel_size=1)

    def forward(self, feats, x0, x1, x2):
        y = self.deconv1(feats)
        y = self.deconv2(self.conv5(torch.cat([y, x2], dim=1)))
        y = self.deconv3(self.conv6(torch.cat([y, x1], dim=1)))
        y = self.conv7(torch.cat([y, x0], dim=1))

        sdf_map = self.sdf_head(y)
        kp_heatmap = self.kp_head(y)
        sdf_map = torch.tanh(sdf_map)
        kp_heatmap = torch.sigmoid(kp_heatmap)
        return sdf_map, kp_heatmap


class SAMHyDAS_Net(nn.Module):
    def __init__(self, nin, n_classes, n_sdf_channels=3, n_kp_channels=2,
                 l_rate=0.001, nG=32, has_dropout=False):
        super().__init__()
        self.encoder = SharedEncoder(nin, nG, has_dropout)
        self.seg_decoder = SegmentationDecoder(n_classes, nG)
        self.reg_decoder = RegressionDecoder(n_sdf_channels, n_kp_channels, nG)

        # Initialize weights
        self.encoder.apply(weights_init)
        self.seg_decoder.apply(weights_init)
        self.reg_decoder.apply(weights_init)

        # Optimizers
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=l_rate)
        self.opt_seg = torch.optim.Adam(self.seg_decoder.parameters(), lr=l_rate)
        self.opt_reg = torch.optim.Adam(self.reg_decoder.parameters(), lr=l_rate/10)
        self.optimizers = [self.opt_enc, self.opt_seg, self.opt_reg]

        # Schedulers
        self.sch_enc = torch.optim.lr_scheduler.ExponentialLR(self.opt_enc, gamma=0.98)
        self.sch_seg = torch.optim.lr_scheduler.ExponentialLR(self.opt_seg, gamma=0.98)
        self.sch_reg = torch.optim.lr_scheduler.ExponentialLR(self.opt_reg, gamma=0.98)
        self.schedulers = [self.sch_enc, self.sch_seg, self.sch_reg]

    def forward(self, x):
        feats, x0, x1, x2 = self.encoder(x)
        seg_logits = self.seg_decoder(feats, x0, x1, x2)
        sdf_map, kp_heatmap = self.reg_decoder(feats, x0, x1, x2)
        return seg_logits, sdf_map, kp_heatmap

    def optimize(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def scheduler_step(self):
        for sch in self.schedulers:
            sch.step()