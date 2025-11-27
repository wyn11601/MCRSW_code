import csv
import os
import random
from collections import Counter
import cvxpy as cp
import networkx as nx
import numpy as np
import ot
import torch
from scipy import interpolate
# from numpy import random
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.special import chebyt

"""
    Structure
    fmri->
        sub_001->
            fmridata.mat
        sub_002->
            fmridata.mat
        sub_003->
            fmridata.mat
        ......
    :return:
    """
# NRR
def NRR(X, y, lambda_val):
    theta = cp.Variable(X.shape[1])

    loss = cp.sum_squares(y - X @ theta)  # \| y - X @ theta \|_2^2
    regularization = lambda_val * cp.norm(theta, 2)  # \ell_2 正则化

    # 构建优化问题
    problem = cp.Problem(cp.Minimize(loss + regularization))
    problem.solve()  # 求解
    return theta.value


def create_DFCN_with_NRR(fmri, num_window, alpha, lambda_val):
    ROI_count, T = fmri.shape
    win_length = T // num_window
    dynamic_networks = []

    for win_idx in range(num_window):
        # 提取当前时间窗口的 fMRI 数据
        window = fmri[:, win_length * win_idx:win_length * (win_idx + 1)]
        sparse_network = np.zeros((ROI_count, ROI_count))

        for n in range(ROI_count):
            # 构建特征矩阵 X 和目标向量 y
            X = np.delete(window, n, axis=0).T  # 去掉第 n 个 ROI
            y = window[n, :]  # 目标向量
            theta_n = NRR(X, y, lambda_val)
            for idx, value in enumerate(theta_n):
                roi_idx = idx if idx < n else idx + 1  # 跳过 n 自己
                sparse_network[n, roi_idx] = value

        # 阈值化稀疏网络
        sparse_network = np.abs(sparse_network)
        dynamic_networks.append(sparse_network)

    return np.array(dynamic_networks)
