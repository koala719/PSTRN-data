# -*- coding: UTF-8 -*-

# change to your path
import sys
sys.path.insert(0, './PSTRN_id640_code_and_data')

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import os
import time
import argparse

from validation.config import proj_cfg

import torch

import torcherry as tc
from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc
from torcherry.utils.util import set_env_seed

from model.cls_rnn_feature import ClassifierFeature
from model.cls_rnn_feature_tr import ClassifierFeatureTR

Models = dict(
    ClassifierFeature=ClassifierFeature,
    ClassifierFeatureTR=ClassifierFeatureTR,
)

def num_para_calcular(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print(r"Total Params：" + str(k))
    return (k)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_env_seed(args.seed, use_cuda)

    dataset_path = os.path.join(proj_cfg.root_path, args.dataset)
    save_path = os.path.join(proj_cfg.save_root, "hmdb51_feature", args.model, args.rank, args.init,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)

    ranks = list(map(lambda x: int(x), args.rank.split(",")))

    if args.model in Models:
        model = Models[args.model](use_cuda=use_cuda, num_class=51, ranks=ranks,
                                   init=args.init, dataset_path=dataset_path, seed=args.seed, dropih=args.dropih)
    else:
        raise ValueError("model %s is not existed!" % args.model)
    num_para_calcular(model)

    training_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc()]

    runner = tc.Runner(use_cuda)
    runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               continual_train_model_dir=args.continual_train_model_dir, pre_train_model_path=args.pre_train_model_path,
               record_setting="%s" % args,
               )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on HMDB51 Feature")

    parser.add_argument("--dataset", type=str, default='/home/drl/lnn/PSTRN_id640_code_and_data/validation/datasets/HMDB51/data_feature_pkls/')
    parser.add_argument("--model", type=str, default="ClassifierFeatureTR")
    parser.add_argument("--init", type=str, default="truncguass_3")
    parser.add_argument("--rank", type=str, default="45,42,36,42")
    parser.add_argument("--dropih", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)
