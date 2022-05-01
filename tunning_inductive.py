from __future__ import print_function

import yaml
import argparse
from rich import print
import tempfile
from datetime import datetime
import time
import json

from src import util
import train_inductive


def main(args):
    tmp = tempfile.NamedTemporaryFile()
    base_config_file = open(args.config_path, 'r')
    config = yaml.load(base_config_file, Loader=yaml.FullLoader)
    config_path = f"{tmp.name}.yml"

    # log file
    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # log_path = f"tunning_{t}.json"
    # fh = open(log_path, "a+")


    # hyperparameters
    embbeding_size_list = [128, ]
    lr_list = [1e-3, ]
    num_points_list = [2048, ]
    batch_size_list = [args.bs, ]
    n_redundant = 10

    n_sets = len(embbeding_size_list) * len(lr_list) * len(num_points_list) * len(batch_size_list) * n_redundant
    count = 0
    for embbeding_size in embbeding_size_list:
        for num_points in num_points_list:
            for batch_size in batch_size_list:
                for lr in lr_list:
                    for i in range(n_redundant):
                        print(f"Training model {count}/{n_sets} ...")
                        # modify config
                        config["lr"] = lr
                        config["num_points"] = num_points
                        config["batch_size"] = batch_size
                        config["embbeding_size"] = embbeding_size
                        config["alpha_sceneaug"] = args.alpha_sceneaug
                        util.write_config(config=config, yml_path=config_path)
                        args.config_path = config_path
                        result = train_inductive.main(args)

                        # print("result", result)
                        if "device" in result: del result["device"]
                        # json.dump(result, fh)
                        # fh.write("\n")
                        count+=1

    # fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', type=str, 
                                          default='ModelNet40', 
                                          choices=['ModelNet40'], 
                                          help='name of dataset i.e. ModelNet')
    parser.add_argument('--dataset_eval', type=str, 
                                          default='ModelNet10', 
                                          choices=['ModelNet10', 'ScanObjectNN'], 
                                          help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
    parser.add_argument('--backbone', type=str, 
                                      default='PointNet', 
                                      choices=['EdgeConv', 'PointAugment', 'PointConv', 'PointNet'], 
                                      help='name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet')
    parser.add_argument('--textencoder', type=str, 
                                      default='bert', 
                                      choices=['bert', 'glove', 'w2v'])
    parser.add_argument('--config_path', type=str, required=True, help='configuration path')
    parser.add_argument('--alpha_sceneaug', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--pbar', action='store_true')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--within', action='store_true')
    parser.add_argument('--noprompt', action='store_true')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=4)
    args = parser.parse_args()

    main(args)
