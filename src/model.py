# src/model.py

import torch
import torch.nn as nn
from typing import Tuple

class HybridSN_BN(nn.Module):
    """
    一个混合了3D和2D卷积的神经网络模型，用于高光谱图像分类。
    在原始HybridSN模型的基础上，为每个卷积层后添加了批量归一化(Batch Normalization)层，
    以加速训练收敛并提升模型性能。
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        """
        初始化网络结构。

        Args:
            in_channels (int): 输入通道数，对于高光谱数据块通常为1。
            out_channels (int): 输出通道数，等于分类任务的类别总数。
        """
        super(HybridSN_BN, self).__init__()
        
        # 3D卷积部分：用于同时提取空间和光谱特征
        self.conv3d_features = nn.Sequential(
            # 第一个3D卷积层
            nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            
            # 第二个3D卷积层
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 第三个3D卷积层
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # 2D卷积部分：在3D卷积提取的特征基础上，进一步学习更抽象的2D空间特征
        self.conv2d_features = nn.Sequential(
            # 2D卷积层的输入通道数是3D卷积最后输出的通道数与光谱维度的乘积
            # 经过3个3D卷积后，光谱维度从30变为 (30-7+1)-5+1-3+1 = 18
            # 输入通道为 32(3D输出通道) * 18(剩余光谱维度) = 576
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 全连接分类器部分
        self.classifier = nn.Sequential(
            # 展平后的特征维度是 64(2D输出通道) * 17*17 (patch在2D卷积后的尺寸)
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播路径。

        Args:
            x (torch.Tensor): 输入的张量，形状为 (N, C, D, H, W)，
                              其中 N=batch_size, C=1, D=光谱维度, H=patch高, W=patch宽。

        Returns:
            torch.Tensor: 模型的输出，形状为 (N, num_classes)。
        """
        # 经过3D卷积层
        x = self.conv3d_features(x)
        
        # 将3D卷积的输出重塑为2D卷积的输入格式
        # (N, 32, 18, 19, 19) -> (N, 32*18, 19, 19)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        
        # 经过2D卷积层
        x = self.conv2d_features(x)
        
        # 展平特征以输入全连接层
        # (N, 64, 17, 17) -> (N, 64*17*17)
        x = x.view(x.size(0), -1)
        
        # 经过分类器
        x = self.classifier(x)
        return x