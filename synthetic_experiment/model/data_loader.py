# -*- coding: UTF-8 -*-

import os
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset

import pickle


class CusDataset(Dataset):
    def __init__(self, data_path):
        super(CusDataset, self).__init__()
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.x = torch.tensor(data["x"])
        self.y = torch.tensor(data["y"])

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
