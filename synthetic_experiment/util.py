# -*- coding: UTF-8 -*-

import collections

import torch
import torch.nn as nn

from torcherry import CherryModule
from model.layers.tr_tensordot_re import TensorRingConvolution, TensorRingLinear


def load_weight_share(is_multi_gpus, use_gpu, model, chepoint_path):
    """
    Warning: Do not set model to parallel
    """
    def generate_state_dict(device, original_dict: dict, target_dict: dict, is_original_parallel=False, is_target_parallel=False):
        if is_original_parallel:
            original_predix = "module."
        else:
            original_predix = ""

        # if is_target_parallel:
        #     target_predix = "module."
        # else:
        #     target_predix = ""

        res_dict = collections.OrderedDict()
        for k, v in target_dict.items():
            # res_k = target_predix + k
            res_k = k
            target_shape = v.shape

            original_v = original_dict[original_predix + k]
            res_v = original_v.to(device)

            for index, shape in enumerate(target_shape):
                res_v = torch.index_select(res_v, index, torch.arange(shape).to(device))
            res_dict[res_k] = res_v

        return res_dict

    assert isinstance(model, CherryModule), "The model must be CherryModule, not a %s!" % type(model)

    device = torch.device("cuda:0" if use_gpu else "cpu")
    pretrained_dict = torch.load(chepoint_path)

    is_checkpoint_parallel = list(pretrained_dict.keys())[0].startswith('module.')

    if is_multi_gpus and use_gpu:
        print("DataParallel...")
        shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), True, True)
        model.load_state_dict(shared_dict)
        for one in model.modules():
            if isinstance(one, TensorRingConvolution):
                one.adjust_parameters_ours_std_res_relu()
            if isinstance(one, TensorRingLinear):
                one.adjust_parameters_ours_std_linear()
        model = nn.DataParallel(model)
    else:
        if is_checkpoint_parallel:
            shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), True, False)
            model.load_state_dict(shared_dict)
        else:
            shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), False, False)
            model.load_state_dict(shared_dict)

    return model




def load_weight_share_without_bn(is_multi_gpus, use_gpu, model, chepoint_path):
    """
    Warning: Do not set model to parallel
    """
    def generate_state_dict(device, original_dict: dict, target_dict: dict, is_original_parallel=False, is_target_parallel=False):
        if is_original_parallel:
            original_predix = "module."
        else:
            original_predix = ""

        # if is_target_parallel:
        #     target_predix = "module."
        # else:
        #     target_predix = ""

        res_dict = collections.OrderedDict()
        for k, v in target_dict.items():
            if "bn" not in k:
                # res_k = target_predix + k
                res_k = k
                target_shape = v.shape

                original_v = original_dict[original_predix + k]
                res_v = original_v.to(device)

                for index, shape in enumerate(target_shape):
                    res_v = torch.index_select(res_v, index, torch.arange(shape).to(device))
                res_dict[res_k] = res_v

        return res_dict

    assert isinstance(model, CherryModule), "The model must be CherryModule, not a %s!" % type(model)

    device = torch.device("cuda:0" if use_gpu else "cpu")
    pretrained_dict = torch.load(chepoint_path)

    is_checkpoint_parallel = list(pretrained_dict.keys())[0].startswith('module.')

    if is_multi_gpus and use_gpu:
        print("DataParallel...")
        shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), True, True)
        model.load_state_dict(shared_dict, strict=False)
        for one in model.modules():
            if isinstance(one, TensorRingConvolution):
                one.adjust_parameters_ours_std_res_relu()
            if isinstance(one, TensorRingLinear):
                one.adjust_parameters_ours_std_linear()
        model = nn.DataParallel(model)
    else:
        if is_checkpoint_parallel:
            shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), True, False)
            model.load_state_dict(shared_dict, strict=False)
        else:
            shared_dict = generate_state_dict(device, pretrained_dict, model.state_dict(), False, False)
            model.load_state_dict(shared_dict, strict=False)

    return model