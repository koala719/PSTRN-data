# -*- coding: UTF-8 -*-

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .layers.tr_tensordot_re import TensorRingLinear

import cus_torcherry as tc

from .data_loader import CusDataset


class mlp_tr(tc.CherryModule):
    def __init__(self, dataset_path, ranks, var, seed=233):
        super(mlp_tr, self).__init__()
        self.data_path = dataset_path
        self.var = var
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.fc = TensorRingLinear(144, 144, [12, 12], [12, 12], [ranks[0], ranks[1], ranks[2], ranks[3]], bias=False, init="ours_linear")

    def forward(self, x):
        x = self.fc(x)

        return x

    def tc_train_step(self, model, data, target):
        output_logits = model(data)
        loss = F.mse_loss(output_logits, target)
        return loss

    def tc_val_step(self, model, data, target):
        output_logits = model(data)
        loss = F.mse_loss(output_logits, target)
        return output_logits, loss

    def tc_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=5e-4)

    def tc_lr_schedule(self, optimizer):
        return MultiStepLR(optimizer, [30, 60, 90], 0.1)

    def tc_train_loader(self):
        self.train_loader_type = "torchvision"
        return DataLoader(CusDataset(os.path.join(self.data_path, "dataset", "var%.2f" % self.var, "data.pkl")), batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=0)

    def tc_val_loader(self):
        self.val_loader_type = "torchvision"
        return DataLoader(CusDataset(os.path.join(self.data_path, "dataset_valid", "var%.2f" % self.var, "data_valid.pkl")), batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=0)

    def tc_test_loader(self):
        self.val_loader_type = "torchvision"
        return DataLoader(CusDataset(os.path.join(self.data_path, "dataset_valid", "var%.2f" % self.var, "data_valid.pkl")), batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=0)


def mlp_tr_r2(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[2, 2, 2, 2], var=var)


def mlp_tr_r3(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[3, 3, 3, 3], var=var)

def mlp_tr_r4(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[4, 4, 4, 4], var=var)


def mlp_tr_r5(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[5, 5, 5, 5], var=var)


def mlp_tr_r6(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 6, 6, 6], var=var)


def mlp_tr_r7(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[7, 7, 7, 7], var=var)


def mlp_tr_r8(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[8, 8, 8, 8], var=var)


def mlp_tr_r9(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, var=var, seed=seed, ranks=[9, 9, 9, 9])


def mlp_tr_r10(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, var=var, seed=seed, ranks=[10, 10, 10, 10])


def mlp_tr_r15(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, var=var, seed=seed, ranks=[15, 15, 15, 15])


def mlp_tr_r20(dataset_path, var, seed=233):
    return mlp_tr(dataset_path, var=var, seed=seed, ranks=[20, 20, 20, 20])


def mlp_tr_research(dataset_path, var, seed=233, rank=None):
    return mlp_tr(dataset_path, var=var, seed=seed, ranks=[rank[0], rank[1], rank[2], rank[3]])