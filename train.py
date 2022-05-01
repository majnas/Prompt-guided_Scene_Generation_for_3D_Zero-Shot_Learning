from __future__ import print_function

import os
from numpy.core.fromnumeric import partition
import torch
# from torch._C import T
# import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import yaml
from rich import print

# import scipy.io
# import torch.optim as optim
# from src.models import *
# from src.loss import *
# from src.models import *
# from src.util import *
# from src.datautil import DataUtil

from src import models
from src import dataset
from src import visualization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    dataset_train = args.dataset_train
    backbone = args.backbone
    workers = int(args.workers)

    config_file = open(args.config_path, 'r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config["dataset_train"] = dataset_train
    config["backbone"] = backbone
    config["workers"] = workers
    config["device"] = device
    print(config)

    # get train/test datasets part
    train_ds, test_ds = dataset.get_dataset(dataset_train=dataset_train, dataset_dir=config["dataset_dir"])

    # create dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    params = {"k": 40, "feature_transform": True, "device": device}
    kwargs = {**config, **params}
    classifier = models.get_classifier(**kwargs)
    classifier.train(train_dl=train_dl, test_dl=test_dl, n_epochs=config["epoch"])
    print(f"Best model accuracy: {classifier.test_acc_best}")




    # for batch_points, batch_lables in train_dl:
    #     print(batch_points.shape, batch_lables)

        # batch_points_np = batch_points.numpy()
        # batch_lables_np = batch_lables.numpy()
        # for points, label in zip(batch_points_np, batch_lables_np):
        #     obj_path = os.path.join(LOG_DIR, f"object_{label[0]}.obj")
        #     visualization.export_color_point_cloud(points=points, obj_path=obj_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN', 'McGill'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
    parser.add_argument('--backbone', type=str, default='PointConv', choices=['EdgeConv', 'PointAugment', 'PointConv', 'PointNet'], help='name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet')
    # parser.add_argument('--method', type=str, default='ours', choices=['ours', 'baseline'], help='name of method i.e. ours, baseline')
    parser.add_argument('--config_path', type=str, required=True, help='configuration path')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=4)
    args = parser.parse_args()

    main(args)