# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import torcherry as tc

from .data_loader import CusDataset

class mlp(tc.CherryModule):
    def __init__(self, dataset_path, seed=233):
        super(mlp, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.fc = nn.Linear(144, 144, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

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
