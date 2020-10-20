# -*- coding: UTF-8 -*-


from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import os
import time
import argparse

import yaml

from config import proj_cfg

import torch

import cus_torcherry as tc
from cus_torcherry.utils.metric import MetricAccuracy, MetricLoss
from cus_torcherry.utils.checkpoint import CheckLowestLoss
from cus_torcherry.utils.util import set_env_seed

from model.cus_mlp_tr import mlp_tr

Models = dict(
    mlp_tr=mlp_tr,
)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_env_seed(args.seed, use_cuda)

    dataset_path = os.path.join(proj_cfg.root_path, args.dataset)
    save_path = os.path.join(proj_cfg.save_root, "enum_valid", "var%.2f" % args.var, args.rank,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)

    ranks = list(map(lambda x: int(x), args.rank.split(",")))

    if args.model in Models:
        model = Models[args.model](dataset_path=dataset_path, seed=args.seed, ranks=ranks, var=args.var)
    else:
        raise ValueError("model %s is not existed!" % args.model)

    training_metrics = [MetricLoss()]
    # training_metrics = None
    valing_metrics = [MetricLoss()]
    # valing_metrics = None
    check_metrics = [CheckLowestLoss()]
    # check_metrics = None

    runner = tc.Runner(use_cuda)
    res_dict = runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               continual_train_model_dir=args.continual_train_model_dir, pre_train_model_path=args.pre_train_model_path,
               record_setting="%s" % args,
               # load_weight_func=load_weight_share_without_bn,
               )

    with open(os.path.join(save_path, "info_record.yml"), "w", encoding="utf-8") as f:
        save_dict = dict(
            rank=ranks,
            loss=res_dict["loss"]
        )
        yaml.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on MLP")

    parser.add_argument("--dataset", type=str, default='./generate_data')
    parser.add_argument("--model", type=str, default="mlp_tr")
    parser.add_argument('--var', type=float, default=0.01, help='var of the noise')
    parser.add_argument("--rank", type=str, default="2,2,2,2")

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)
