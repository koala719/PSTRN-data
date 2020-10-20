# -*- coding: UTF-8 -*-

import os

import time
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from tqdm import tqdm

from . import CherryModule

from .utils.util import load_model, ContinualTrain, create_nonexistent_folder


class Runner(object):
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda
        self.gpu_num = torch.cuda.device_count()
        self.multi_gpus = self.gpu_num > 1
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def fit(self, model: CherryModule, train_loader=None, optimizer=None, lr_schedule=None, train_step_func=None,
            save_path="./save", use_tensorboard=True, train_callbacks=None, val_callbacks=None, val_loader=None,
            continual_train_model_dir=None, record_setting: str = None, pre_train_model_path=None, train_epochs=0,
            checkpoint_callbacks=None, val_step_func=None, model_dir=None, log_dir=None, tbflush_feq=3,
            save_ori_model_flag=True, load_weight_func=None):

        self.model = model
        self.model.to(self.device)

        self.load_weight = load_weight_func if load_weight_func else load_model

        self.optimizer = optimizer if optimizer else model.tc_optimizer()
        self.lr_schedule = lr_schedule(self.optimizer) if lr_schedule else model.tc_lr_schedule(self.optimizer)
        self.train_step_func = train_step_func if train_step_func else model.tc_train_step
        self.val_step_func = val_step_func if val_step_func else model.tc_val_step
        self.train_loader = train_loader if train_loader else model.tc_train_loader()
        self.val_loader = val_loader if val_loader else model.tc_val_loader()

        # ----------------- Continual Training -------------------------
        # Load Model, Optimizer
        # Set model_dir, log_dir, and start_epoch

        self.lowest_loss = math.inf
        self.train_start_epoch = 0

        # ------------------- Set Model-Save Path ----------------------

        self.model_root = save_path
        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)

        # ------------------- Save Running Setting ----------------------
        if record_setting:
            with open(os.path.join(self.model_root, "running_setting.txt"), "w") as f:
                f.write(record_setting)

        # ------------------- Set Models-Save Path ----------------------

        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(self.model_root, "checkpoint")
        create_nonexistent_folder(self.model_dir)

        # ------------------- Set Logs-Save Path ----------------------

        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(self.model_root, "logs")

        create_nonexistent_folder(self.log_dir)

        # Load Pretrained Model
        if pre_train_model_path:
            print("Loading Pre-trained Model...")
            self.model = self.load_weight(self.multi_gpus, self.use_cuda, self.model, pre_train_model_path)
        else:
            if self.multi_gpus and self.use_cuda:
                print("DataParallel...")
                self.model = nn.DataParallel(self.model)

            if save_ori_model_flag:
                print("Saving original model...")
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_dir, 'model-nn-ori.pt'))

        # ------------------- Set Summary Writer ----------------------
        if use_tensorboard:
            self.summary_writer = SummaryWriter(logdir=self.log_dir)
        else:
            self.summary_writer = None

        # --------------- Training --------------

        for epoch in range(self.train_start_epoch, train_epochs):
            if self.summary_writer:
                self.summary_writer.add_scalar(r"learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            print("\nTraining Epoch: %d/%d" % (epoch, train_epochs - 1), "| Lowest Loss:", self.lowest_loss)
            self.model.train()

            start_time_epoch = time.time()
            with tqdm(total=len(self.train_loader)) as pbar:
                for data_pairs in self.train_loader:
                    self._torchvision_step(pbar, data_pairs)

            # Update Learning Rate
            self.lr_schedule.step()

            end_time_epoch = time.time()
            train_time_epoch = end_time_epoch - start_time_epoch
            if self.summary_writer:
                self.summary_writer.add_scalar("epoch_train_time", train_time_epoch, epoch)

            # Test Training Dataset
            if train_callbacks:
                print("Evaluating Training Data...")
                training_metrics = self._val_by_torch(self.train_loader, train_callbacks)

                res_training = []
                for training_metric in training_metrics:
                    res_training.append(training_metric.get_metric())

                # save logs
                if self.summary_writer:
                    for train_metric in res_training:
                        self.summary_writer.add_scalar(r"train/" + train_metric["metric_name"], train_metric["metric_value"],
                                                  epoch)

            # Test Testing Dataset
            if val_callbacks:
                print("Evaluating Valing Data...")
                valing_metrics = self._val_by_torch(self.val_loader, val_callbacks)

                res_val = []
                for metric in valing_metrics:
                    res_val.append(metric.get_metric())

                for val_metric in res_val:
                    if self.summary_writer:
                        self.summary_writer.add_scalar(r"val/" + val_metric["metric_name"], val_metric["metric_value"], epoch)

                    # Update Test Top_1
                    if val_metric["metric_name"] == "loss":
                        if val_metric["metric_value"] < self.lowest_loss:
                            print("Loss(%f) of Epoch %d is lowest now." % (val_metric["metric_value"], epoch))

                            self.lowest_loss = val_metric["metric_value"]
                            if self.summary_writer:
                                self.summary_writer.add_scalar(r"lowest_loss", self.lowest_loss, epoch)

            # save checkpoint
            if checkpoint_callbacks:
                for checkpoint_callback in checkpoint_callbacks:
                    checkpoint_callback.check2save(self, epoch)

            # flush Tensorboard
            if self.summary_writer:
                if epoch % tbflush_feq == 0:
                    self.summary_writer.flush()

        if self.summary_writer:
            self.summary_writer.close()
        print("Run Over!")
        return {"loss": self.lowest_loss}

    def test(self, model, model_path, test_callbacks=None, test_loader=None, test_step_func=None):
        self.model = model
        self.model.to(self.device)

        print("Loading Pre-trained Model...")
        self.model = load_model(self.multi_gpus, self.use_cuda, self.model, model_path)

        if test_loader:
            self.test_loader = test_loader
        else:
            self.test_loader = model.tc_test_loader()

        if test_step_func:
            self.test_step_func = test_step_func
        else:
            self.test_step_func = model.tc_test_step

        # Test Testing Dataset
        if test_callbacks:
            print("Testing Model...")
            testing_metrics = self._test_by_torch(self.test_loader, test_callbacks)

            res_test = []
            for metric in testing_metrics:
                res_test.append(metric.get_metric())

            for test_metric in res_test:
                print(test_metric)

    def _torchvision_step(self, pbar, data_pairs):
        data = data_pairs[0].to(self.device, non_blocking=True)
        target = data_pairs[1].to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        loss = self.train_step_func(self.model, data, target)

        loss.backward()
        self.optimizer.step()

        pbar.update(1)

    def _val_by_torch(self, loader, metrics):
        metrics_ = copy.deepcopy(metrics)

        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for data_pair in loader:
                    data = data_pair[0].to(self.device, non_blocking=True)
                    target = data_pair[1].to(self.device, non_blocking=True)
                    output_logits, loss = self.val_step_func(self.model, data, target)

                    for metric in metrics_:
                        metric.add_metric_record(output_logits, target, loss)

                    pbar.update(1)

        return metrics_

    def _test_by_torch(self, loader, metrics):
        metrics_ = copy.deepcopy(metrics)

        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for data_pair in loader:
                    data = data_pair[0].to(self.device, non_blocking=True)
                    target = data_pair[1].to(self.device, non_blocking=True)
                    output_logits, loss = self.test_step_func(self.model, data, target)

                    for metric in metrics_:
                        metric.add_metric_record(output_logits, target, loss)

                    pbar.update(1)

        return metrics_
