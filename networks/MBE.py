import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeDetector(nn.Module):
    """
    Sobel 边缘检测器，用于提取图像的水平和垂直梯度。
    """
    def __init__(self, in_channels):
        super(SobelEdgeDetector, self).__init__()
        # 定义 Sobel 卷积核
        sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # 注册为卷积权重
        self.sobel_x = nn.Parameter(sobel_x.repeat(1, in_channels, 1, 1), requires_grad=True)
        self.sobel_y = nn.Parameter(sobel_y.repeat(1, in_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        """
        前向传播：
        1. 使用 Sobel 卷积核计算水平和垂直梯度。
        2. 返回梯度的幅值（可选）。
        """
        # 计算水平和垂直梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # 计算梯度幅值（可选）
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude


class EDM(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2):
        super(EDM, self).__init__()
        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            SobelEdgeDetector(in_c),
            nn.Conv2d(in_channels=1, out_channels=in_c, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.conv_identity = nn.Sequential(
            SobelEdgeDetector(in_c),
            nn.Conv2d(in_channels=1, out_channels=in_c, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.conv_down = nn.Sequential(
            SobelEdgeDetector(in_c),
            nn.Conv2d(in_channels=1, out_channels=in_c, kernel_size=(3, 3), stride=2, padding=1),
            nn.Upsample(scale_factor=scale_factor)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=scale_factor)
        self.bn = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3*in_c, kernel_size=(1, 1), out_channels=out_c)

    def forward(self, x):
        x1 = self.conv_up(x)
        x2 = self.conv_identity(x)
        x3 = self.conv_down(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.act1(self.bn(self.conv1(x)))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class GEM(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2):
        super(GEM, self).__init__()
        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.conv_identity = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=(3, 3), stride=2, padding=1),
            nn.Upsample(scale_factor=scale_factor)
        )

        self.up = nn.Upsample(scale_factor=scale_factor)
        self.bn = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3*in_c, kernel_size=(1, 1), out_channels=out_c)

        self.cbam = CBAM(in_channels=in_c)

    def forward(self, x):
        temp = x
        x1 = self.conv_up(x)
        x2 = self.conv_identity(x)
        x3 = self.conv_down(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.act1(self.bn(self.conv1(x)))
        x = self.cbam(x)
        x = x + temp
        return x