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

from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc


from validation.config import proj_cfg

from model.lenet import LeNet5_Fashion_Mnist
from model.lenet_tensor import *
from model.lenet_tensor_mnist import *
from torcherry.utils.util import set_env_seed



Models = dict(
    LeNet5_Fashion_Mnist=LeNet5_Fashion_Mnist,
    LeNet5_Tensor_Mnist_Rsearch=LeNet5_Tensor_Mnist_Rsearch,
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

    save_path = os.path.join(proj_cfg.save_root, args.init,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)
    ranks = list(map(lambda x: int(x), args.rank.split(",")))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    runner = tc.Runner(use_cuda)  # Must be setted before defining models

    if args.model in Models:
        model = Models[args.model](dataset_path=dataset_path, num_classes=10, init=args.init, seed=args.seed,
                                   rank=ranks)
    else:
        raise ValueError("model %s is not existed!" % args.model)

    num_para_calcular(model)

    training_metrics = [MetricAccuracy(1), MetricLoss()]
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc()]
    #
    runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
                           val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
                           continual_train_model_dir=args.continual_train_model_dir,
                           pre_train_model_path=args.pre_train_model_path
                           )

    print("Run Over!")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on FASHION MNIST")

    parser.add_argument("--dataset", type=str, default='./datasets/')
    parser.add_argument("--model", type=str, default="LeNet5_Tensor_Mnist_Rsearch")
    parser.add_argument("--init", type=str, default="ours_conv_relu")

    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')
    parser.add_argument("--rank", type=str, default="6,20,14,8,12,20,2,20,16,16,20,12,12,8,6,26,8,2,6,20")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)