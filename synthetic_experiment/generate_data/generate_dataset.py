# -*- coding: UTF-8 -*-


from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import os

import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

file_path = os.path.dirname(os.path.realpath(__file__))

class mlp(nn.Module):
    def __init__(self, groundtruth_path):
        super(mlp, self).__init__()
        with open(groundtruth_path, "rb") as f:
            groundtruth = pickle.load(f)

        # self.fc_param = nn.Parameter(torch.from_numpy(groundtruth["fc"]))
        self.fc_param = nn.Parameter(torch.tensor(groundtruth["fc"]))

    def forward(self, x):
        x = x.matmul(self.fc_param)

        return x


if __name__ == '__main__':
    num_sample = 4000
    input_size = 144

    groundtruth_path = os.path.join(file_path, "groundtruth", "groundtruth.pkl")
    cus_mlp = mlp(groundtruth_path)
    cus_mlp.cuda()

    x_var = 0.5
    x = torch.randn(num_sample, input_size).cuda()
    nn.init.normal_(x, 0, np.sqrt(x_var))
    with torch.no_grad():
        y = cus_mlp(x)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    dataset_folder = os.path.join(file_path, "dataset")
    for var in np.linspace(0.01, 0.15, 15):
        var_folder = os.path.join(dataset_folder, "var%.2f" % var)
        x = x + np.random.normal(0, np.sqrt(var), (num_sample, input_size)).astype(np.float32)
        data_dict = dict(
            x=x,
            y=y
        )
        if not os.path.exists(var_folder):
            os.makedirs(var_folder)
        with open(os.path.join(var_folder, "data.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
