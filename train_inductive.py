from __future__ import print_function

import os
import yaml
import numpy as np
import argparse
from rich import print
from datetime import datetime
import torch
import torch.nn.parallel
import torch.utils.data

from src import models
from src import dataset
from src import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    config_file = open(args.config_path, 'r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config["dataset_train"] = args.dataset_train
    config["dataset_eval"] = args.dataset_eval
    config["backbone"] = args.backbone
    config["workers"] = args.workers
    config["verbose"] = args.verbose    
    config["pbar"] = args.pbar 
    config["within"] = args.within
    config["noprompt"] = args.noprompt
    config["device"] = device
    config["alpha_sceneaug"] = args.alpha_sceneaug
    config["batch_size"] = args.bs
    config["textencoder"] = args.textencoder
    config["text_embeddings_path"] = config[f"{args.textencoder}_text_embeddings_path"]
    config["text_embedding_size"] = config[f"{args.textencoder}_text_embedding_size"]

    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logs_dir = os.path.dirname(os.path.join(config['logs_dir'], ''))
    checkpoints_dir = os.path.dirname(os.path.join(config['checkpoints_dir'], ''))
    alpha_sceneaug_str = f"_alpha={config['alpha_sceneaug']}"
    config["logs_dir"] = f"{logs_dir}_inductive{alpha_sceneaug_str}_{t}"
    config["checkpoints_dir"] = f"{checkpoints_dir}_inductive{alpha_sceneaug_str}_{t}"
    os.makedirs(config["logs_dir"], exist_ok=True)
    os.makedirs(config["checkpoints_dir"], exist_ok=True)
    util.write_config(config)
    print(config)
    print(f"[ZSL] >> scene_aug: {args.alpha_sceneaug}")
    print(f"[ZSL] >> verbose: {args.verbose}")


    # get text embeddings to test model
    # get train/test datasets parts
    ds = dataset.SemanticDataset(**config)
    print(f"[ZSL] >> zs_text_embeddings: {ds.zs_text_embeddings.cpu().numpy().shape}")
    print(f"[ZSL] >> gzs_text_embeddings: {ds.gzs_text_embeddings.cpu().numpy().shape}")
    # print(ds.test_seen_noaug_ds.n_samples)

    # create dataloaders
    train_seen_dl = torch.utils.data.DataLoader(ds.train_seen_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    test_unseen_noaug_mapped_dl = torch.utils.data.DataLoader(ds.test_unseen_noaug_mapped_ds, batch_size=config["batch_size"], shuffle=False)
    test_seen_noaug_dl = torch.utils.data.DataLoader(ds.test_seen_noaug_ds, batch_size=config["batch_size"], shuffle=False)
    test_unseen_noaug_dl = torch.utils.data.DataLoader(ds.test_unseen_noaug_ds, batch_size=config["batch_size"], shuffle=False)

    params = {"zs_test_texts": ds.zs_test_texts,
              "gzs_test_texts": ds.gzs_test_texts,
              "zs_text_embeddings": ds.zs_text_embeddings,
              "gzs_text_embeddings": ds.gzs_text_embeddings,
              "all_class_names": ds.train_seen_ds.all_class_names
              }


    kwargs = {**config, **params}
    inductive = models.get_inductive(**kwargs)
    # print("inductive", inductive.model)

    inductive.train(train_seen_dl=train_seen_dl, 
                    test_seen_dl=test_seen_noaug_dl,
                    test_unseen_mapped_dl=test_unseen_noaug_mapped_dl, 
                    test_unseen_dl=test_unseen_noaug_dl, 
                    n_epochs=config["epoch"])

    result = dict(zs_test_acc_best=inductive.zs_test_acc_best,
                  gzs_test_sacc_best=inductive.gzs_test_acc_best[0],
                  gzs_test_uacc_best=inductive.gzs_test_acc_best[1],
                  gzs_test_hmacc_best=inductive.gzs_test_acc_best[2])
    print(result)
    ret = {**config, **result}
    return ret




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