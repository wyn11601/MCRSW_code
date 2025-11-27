from torch import sigmoid
from torch_geometric.nn import GATConv
from opt import *
import torch.nn.functional as F
from torch import nn
import torch

class D_GAT(nn.Module):
    def __init__(self):
        super(D_GAT, self).__init__()
        self.num_layers = 4
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs1.append(GATConv(in_channels=512, out_channels=20, heads=2))
        self.convs2.append(GATConv(in_channels=512, out_channels=20, heads=2))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.convs2.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.convs2.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.convs2.append(GATConv(in_channels=20, out_channels=20, heads=2))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(80, 2)

        # Set initial weights to speed up the training process.
        self.weights1 = torch.nn.Parameter(torch.empty(4).fill_(0.8))
        self.weights2 = torch.nn.Parameter(torch.empty(4).fill_(0.2))

        self.a = torch.nn.Parameter(torch.Tensor(20, 1))

    def reset_parameters(self):
        for conv in self.convs1:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        self.a.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, same_index, diff_index):
        x = features

        # Graph transformer and information aggregation layers.
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[0](x, same_index)
        x2 = self.convs2[0](x, diff_index)
        weight1 = self.weights1[0] / (self.weights1[0] + self.weights2[0])
        weight2 = self.weights2[0] / (self.weights1[0] + self.weights2[0])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[0](x)
        x = F.leaky_relu(x, inplace=True)
        fc = x
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[1](x, same_index)
        x2 = self.convs2[1](x, diff_index)
        weight1 = self.weights1[1] / (self.weights1[1] + self.weights2[1])
        weight2 = self.weights2[1] / (self.weights1[1] + self.weights2[1])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[1](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[2](x, same_index)
        x2 = self.convs2[2](x, diff_index)
        weight1 = self.weights1[2] / (self.weights1[2] + self.weights2[2])
        weight2 = self.weights2[2] / (self.weights1[2] + self.weights2[2])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[2](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[3](x, same_index)
        x2 = self.convs2[3](x, diff_index)
        weight1 = self.weights1[3] / (self.weights1[3] + self.weights2[3])
        weight2 = self.weights2[3] / (self.weights1[3] + self.weights2[3])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[3](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)
        x = self.out_fc(fc)
        return x