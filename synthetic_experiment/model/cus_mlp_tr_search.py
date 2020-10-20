# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .layers.tr_tensordot_re import TensorRingLinear

import cus_torcherry as tc

from .data_loader import CusDataset


class mlp_tr(tc.CherryModule):
    def __init__(self, dataset_path, ranks, seed=233):
        super(mlp_tr, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.fc = TensorRingLinear(144, 144, [12, 12], [12, 4, 3], [ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[5], ranks[6], ranks[7]], bias=False, init="ours_linear")

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
        return DataLoader(CusDataset(self.data_path), batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=0)

    def tc_val_loader(self):
        self.val_loader_type = "torchvision"
        return DataLoader(CusDataset(self.data_path), batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=0)


def mlp_tr_v1(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 12, 15, 15, 3, 6, 9, 9])


def mlp_tr_v2(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[15, 15, 3, 9])


def mlp_tr_v3(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 15, 3, 3])


def mlp_tr_v4(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[9, 6, 12, 9])


def mlp_tr_v5(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 12, 3, 9])


def mlp_tr_v6(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[3, 15, 6, 9])


def mlp_tr_v7(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[9, 15, 15, 12, 15, 15, 12, 9])


def mlp_tr_v8(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[3, 9, 9, 15])


def mlp_tr_v9(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[3, 15, 15, 15])


def mlp_tr_v10(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 15, 15, 15])


def mlp_tr_v11(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[12, 15, 6, 15])


def mlp_tr_v12(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[9, 15, 3, 9])


def mlp_tr_v13(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[6, 15, 12, 15])


def mlp_tr_v14(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[3, 15, 3, 3])


def mlp_tr_v15(dataset_path, seed=233):
    return mlp_tr(dataset_path, seed=seed, ranks=[9, 6, 15, 15, 9, 6, 3, 6])
