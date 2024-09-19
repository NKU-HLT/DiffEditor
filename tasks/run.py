import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    import torch
    import random
    import numpy as np

    randseed=629264
    
    # 设置随机种子
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    # 如果使用GPU，设置cuda的随机种子
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)  # 如果使用多个GPU

    # 确保CuDNN的可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    set_hparams()
    run_task()

