import os
import sys
import shutil
import random
import errno

import torch
import numpy as np
from torch.optim import AdamW


def build_optimizer(args, model):

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return optimizer


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(root_save_path):
    if os.path.exists(root_save_path):
        shutil.rmtree(root_save_path)  # delete output folder
    os.makedirs(root_save_path)  # make new output folder

# 显示某个模型的梯度信息
def show_grad(model):
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 如果参数的梯度不存在，则打印提示信息
        if param.grad is None:
            print(f"{name:<30} ⟶  梯度还没生成")
        # 否则，打印参数名称、形状和梯度均值
        else:
            print(f"{name:<30} | 形状: {tuple(param.grad.shape)} | "
                  f"梯度均值: {param.grad.mean():.4e}")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
