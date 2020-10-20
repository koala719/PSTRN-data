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

from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc
from torcherry.utils.util import set_env_seed


from model.resnet import resnet20, resnet32
from model.resnet_tr_cifar10 import *

Models = dict(
    resnet20_tr_cifar10_rsearch=resnet20_tr_cifar10_rsearch,
    resnet32_tr_cifar10_rsearch=resnet32_tr_cifar10_rsearch
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
    save_path = os.path.join(proj_cfg.save_root, "cifar10", args.model, args.init,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)
    ranks = list(map(lambda x: int(x), args.rank.split(",")))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.model in Models:
        model = Models[args.model](dataset_path=dataset_path, seed=args.seed, Rank=ranks)
    else:
        raise ValueError("model %s is not existed!" % args.model)
    num_para_calcular(model)

    training_metrics = None
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc()]

    runner = tc.Runner(use_cuda)


    runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               continual_train_model_dir=args.continual_train_model_dir, pre_train_model_path=None,
               record_setting="%s" % args,
               load_weight_func=None,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on Cifar10")

    parser.add_argument("--dataset", type=str, default='./datasets/')
    parser.add_argument("--model", type=str, default="resnet32_tr_cifar10_rsearch")
    parser.add_argument("--init", type=str, default="ours_res_relu")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rank", type=str, default="3, 12, 9, 9, 9, 9, 6")

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)
