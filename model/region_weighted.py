from torch import sigmoid
from torch_geometric.nn import TransformerConv
from opt import *
import torch.nn.functional as F
from torch import nn
import torch
class RegionCNNBlock(nn.Module):
    def __init__(self, input_dim, dropout):
        super(RegionCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class RegionalCNNModule(nn.Module):
    """# Since we are in the process of applying for a patent,
        we cannot provide detailed zoning information for each area at the moment.
        However, rest assured that we will complete the code later.
"""
    def __init__(self, input_dim, dropout_prob=0.2):
        super(RegionalCNNModule, self).__init__()
        # 为每个区域分别定义两层 CNN 模块（这里保持输入输出通道数均为3）
        self.region1_conv = RegionCNNBlock(input_dim, dropout_prob)
        self.region2_conv = RegionCNNBlock(input_dim, dropout_prob)
        self.region3_conv = RegionCNNBlock(input_dim, dropout_prob)
        self.region4_conv = RegionCNNBlock(input_dim, dropout_prob)
        # 为每个区域定义一个可学习的标量权重，初始值设为1.0
        self.weight1 = nn.Parameter(torch.zeros(0, 116))#第一个参数为区域权重维度,我们通过0替代
        nn.init.xavier_uniform_(self.weight1, gain=1.414)
        self.weight2 = nn.Parameter(torch.zeros(0, 116))
        nn.init.xavier_uniform_(self.weight2, gain=1.414)
        self.weight3 = nn.Parameter(torch.zeros(0, 116))
        nn.init.xavier_uniform_(self.weight3, gain=1.414)
        self.weight4 = nn.Parameter(torch.zeros(0, 116))
        nn.init.xavier_uniform_(self.weight4, gain=1.414)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        region1_idx = list()  # FR
        region2_idx = list()  # LIR
        region3_idx = sorted(list() + list())  #PCR
        region4_idx = sorted(list() + list())  # SCR
        out = x.clone()
        # region
        x_region1 = x[:, :, region1_idx, :]  #
        processed_r1 = self.region1_conv(x_region1)  #
        processed_r1 = processed_r1 * self.weight1  #
        x_region2 = x[:, :, region2_idx, :]
        processed_r2 = self.region2_conv(x_region2) * self.weight2
        x_region3 = x[:, :, region3_idx, :]
        processed_r3 = self.region3_conv(x_region3) * self.weight3
        x_region4 = x[:, :, region4_idx, :]
        processed_r4 = self.region4_conv(x_region4) * self.weight4
        out[:, :, region1_idx, :] = processed_r1
        out[:, :, region2_idx, :] = processed_r2
        out[:, :, region3_idx, :] = processed_r3
        out[:, :, region4_idx, :] = processed_r4
        out = self.dropout(out)

        return out

