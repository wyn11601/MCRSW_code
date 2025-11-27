from torch import sigmoid
from torch_geometric.nn import TransformerConv
from opt import *
import torch.nn.functional as F
from torch import nn
import torch

class Temporal(nn.Module):
    def __init__(self, w_num, roi_num, feature_d):
        super(Temporal, self).__init__()

        self.w_num = w_num
        self.roi_num = roi_num
        self.feature_d = feature_d
        self.ptn = nn.Parameter(torch.zeros(self.roi_num, 1))
        nn.init.xavier_uniform_(self.ptn.data, gain=1.414)
        self.wn = nn.Parameter(torch.zeros(self.w_num, self.feature_d))
        nn.init.xavier_uniform_(self.wn.data, gain=1.414)
        self.ptd = nn.Parameter(torch.zeros(self.feature_d, 1))
        nn.init.xavier_uniform_(self.ptd.data, gain=1.414)
        self.wd = nn.Parameter(torch.zeros(self.roi_num, self.w_num))
        nn.init.xavier_uniform_(self.wd.data, gain=1.414)
        self.v_e = nn.Parameter(torch.zeros(self.w_num, self.w_num))
        nn.init.xavier_uniform_(self.v_e.data, gain=1.414)

    def forward(self, x):
        a = x.permute(0, 1, 3, 2)
        channel2 = torch.matmul(a, self.ptn)
        channel2 = channel2.reshape(x.shape[0], self.roi_num, self.feature_d)
        channel2 = channel2*self.wn.unsqueeze(0)
        b = x.permute(2, 0, 3, 1)
        ptd = torch.squeeze(self.ptd)
        channel1 = torch.matmul(ptd, b)
        channel1 = channel1.permute(1, 0, 2)
        channel1 = channel1*self.wd.unsqueeze(0)
        dual_channel = torch.matmul(channel1, channel2)
        dual_channel = sigmoid(dual_channel).permute(1, 2, 0)
        dual_channel = torch.matmul(self.v_e, dual_channel)
        s = dual_channel.permute(2, 0, 1)
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp = torch.exp(s)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)
        return S_normalized

class temporal_weighted(nn.Module):
    def __init__(self, w_num, roi_num, feature_d):
        super(temporal_weighted, self).__init__()
        self.w_num = w_num
        self.roi_num = roi_num
        self.feature_d = feature_d
        self.temporal = Temporal(self.w_num, self.roi_num, self.feature_d)
    def forward(self, x):

        temporal_matrix = self.temporal(x)
        x_temporal = reshape_dot(x, temporal_matrix)
        return x_temporal
def reshape_dot(x, TATT):
    outs = torch.matmul((x.permute(0, 2, 3, 1))
                        .reshape(x.shape[0], -1, x.shape[1]), TATT).reshape(-1, x.shape[1],
                                                                            x.shape[2],
                                                                            x.shape[3])
    return outs