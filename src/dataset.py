import torch
import numpy as np
from torch.autograd.grad_mode import F
import torchvision.transforms as T
from torchvision.transforms import autoaugment
import scipy.io as sio

from src import datautil
from src import scenetrans
from src import augmentation

"""
  Data Setup:
         ModelNet40
         +---------+----------+
         |         |          |      
  train  |   seen  |    X     |
         |   30C   |          |
         +---------+----------+
         |         |          |      
  test   |   seen  |  unseen  |
         |   30C   |  10C     |
         +---------+----------+
  >> Train model on ModelNet40_train_seen_30C part
  >> Evaluate model on ModelNet40_test_unseen_10C part in ZSL
  >> Evaluate model on test (ModelNet40_test_seen_30C + ModelNet40_test_unseen_10C) part in GZSL
"""

"""
  Data Setup:
         ModelNet40
         +---------+----------+
         |         |          |      
  train  |   seen  |    X     |
         |   26C   |          |
         +---------+----------+
         |         |          |      
  test   |   seen  |    X     |
         |   26C   |          |
         +---------+----------+

         ScanObjectNN
         +---------+----------+
  train  |    X    |    X     |
         +---------+----------+
         |         |          | 
  test   |    X    |  unseen  |
         |         |  11C     |
         +---------+----------+

  >> Train model on ModelNet40_train_seen_26C part
  >> Evaluate model on ScanObjectNN_test_unseen_11C part in ZSL
  >> Evaluate model on test (ModelNet40_test_seen_26C + ScanObjectNN_test_unseen_11C) part in GZSL
"""


MODELNET40_MODELNET10_SEEN_IDS = [0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39]
MODELNET40_MODELNET10_SEEN_LABEL_MAP = {cidx:idx for (idx,cidx) in enumerate(MODELNET40_MODELNET10_SEEN_IDS)}
MODELNET40_MODELNET10_ZS_UNSEEN_IDS = [1,2,8,12,14,22,23,30,33,35]
MODELNET40_MODELNET10_ZS_UNSEEN_LABEL_MAP = {cidx:idx for (idx,cidx) in enumerate(MODELNET40_MODELNET10_ZS_UNSEEN_IDS)}
MODELNET40_MODELNET10_GZS_SEEN_IDS = [0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39]
# MODELNET40_MODELNET10_GZS_SEEN_LABEL_MAP = {cidx:cidx for cidx in MODELNET40_MODELNET10_SEEN_IDS}
MODELNET40_MODELNET10_GZS_UNSEEN_IDS = [1,2,8,12,14,22,23,30,33,35]
# MODELNET40_MODELNET10_GZS_UNSEEN_LABEL_MAP = {cidx:cidx for cidx in MODELNET40_MODELNET10_GZS_UNSEEN_IDS}


MODELNET40_SCANOBJECTNN_TRAIN_SEEN_IDS = [0,1,5,6,7,9,10,11,15,16,17,18,19,20,21,23,24,25,26,27,28,31,34,36,37,39]
MODELNET40_SCANOBJECTNN_TRAIN_SEEN_LABEL_MAP = {cidx:idx for (idx,cidx) in enumerate(MODELNET40_SCANOBJECTNN_TRAIN_SEEN_IDS)}
MODELNET40_SCANOBJECTNN_ZS_UNSEEN_IDS = [3,4,5,6,7,8,9,10,12,13,14] # map from ScanPbjectNN
MODELNET40_SCANOBJECTNN_ZS_UNSEEN_LABEL_MAP = {cidx:idx for (idx,cidx) in enumerate(MODELNET40_SCANOBJECTNN_ZS_UNSEEN_IDS)}
MODELNET40_SCANOBJECTNN_GZS_SEEN_IDS = [0,1,5,6,7,9,10,11,15,16,17,18,19,20,21,23,24,25,26,27,28,31,34,36,37,39] # map from ModelNet40
MODELNET40_SCANOBJECTNN_GZS_SEEN_LABEL_MAP = {cidx:idx for (idx,cidx) in enumerate(MODELNET40_SCANOBJECTNN_GZS_SEEN_IDS)}
MODELNET40_SCANOBJECTNN_GZS_UNSEEN_IDS = [3,4,5,6,7,8,9,10,12,13,14] # There is 26 seen objects from ModelNet40 + 11 objects from ScanPbjectNN
MODELNET40_SCANOBJECTNN_GZS_UNSEEN_LABEL_MAP = {cidx:26+idx for (idx,cidx) in enumerate(MODELNET40_SCANOBJECTNN_GZS_UNSEEN_IDS)}



ModelNet40_ModelNet10 = dict(SEEN_IDS=MODELNET40_MODELNET10_SEEN_IDS,
                             SEEN_LABEL_MAP=MODELNET40_MODELNET10_SEEN_LABEL_MAP,
                             ZS_UNSEEN_IDS=MODELNET40_MODELNET10_ZS_UNSEEN_IDS,
                             ZS_UNSEEN_LABEL_MAP=MODELNET40_MODELNET10_ZS_UNSEEN_LABEL_MAP,
                             GZS_SEEN_IDS=MODELNET40_MODELNET10_GZS_SEEN_IDS,
                             GZS_SEEN_LABEL_MAP=[],
                             GZS_UNSEEN_IDS=MODELNET40_MODELNET10_GZS_UNSEEN_IDS,
                             GZS_UNSEEN_LABEL_MAP=[])

ModelNet40_ScanObjectNN = dict(SEEN_IDS=MODELNET40_SCANOBJECTNN_TRAIN_SEEN_IDS,
                               SEEN_LABEL_MAP=MODELNET40_SCANOBJECTNN_TRAIN_SEEN_LABEL_MAP,
                               ZS_UNSEEN_IDS=MODELNET40_SCANOBJECTNN_ZS_UNSEEN_IDS,
                               ZS_UNSEEN_LABEL_MAP=MODELNET40_SCANOBJECTNN_ZS_UNSEEN_LABEL_MAP,
                               GZS_SEEN_IDS=MODELNET40_SCANOBJECTNN_GZS_SEEN_IDS,
                               GZS_SEEN_LABEL_MAP=MODELNET40_SCANOBJECTNN_GZS_SEEN_LABEL_MAP,
                               GZS_UNSEEN_IDS=MODELNET40_SCANOBJECTNN_GZS_UNSEEN_IDS,
                               GZS_UNSEEN_LABEL_MAP=MODELNET40_SCANOBJECTNN_GZS_UNSEEN_LABEL_MAP,
                               )

# ZSL_DATASET_INFO = dict(ModelNet40_ModelNet10=ModelNet40_ModelNet10,
#                         ModelNet40_ScanObjectNN=ModelNet40_ScanObjectNN)



def get_dataset(dataset_train: str, dataset_dir: str, part: str = "all"):
    if dataset_train == "ModelNet":
        train_ds = datautil.ModelNet40(dataset_dir=dataset_dir, train=True, part=part, num_points=2048, use_normals=True)
        test_ds = datautil.ModelNet40(dataset_dir=dataset_dir, train=False, part=part, num_points=2048, use_normals=True)
        # print(f"[ZSL] >> Train Samples -------> {train_ds.n_samples}")
        # print(f"[ZSL] >> Test Samples --------> {test_ds.n_samples}")
        # print(train_ds.statistics)
        # print(train_ds.class_names)
        # print(train_ds.n_classes)
    return train_ds, test_ds


def get_transforms(alpha_sceneaug: float = 0.0, within: bool =  False, noprompt: bool = False) -> tuple :
    noaug_trans = scenetrans.OneOf([scenetrans.ThisIsAObject(p=1.0, noprompt=noprompt)], 
                                    p=alpha_sceneaug)
    aug_trans = scenetrans.OneOf([scenetrans.ThisIsAObject(p=1.0, noprompt=noprompt),
                                  scenetrans.ABigObject(p=1.0, noprompt=noprompt),
                                  scenetrans.ASmallObject(p=1.0, noprompt=noprompt),
                                  scenetrans.TwoObjects(p=1.0, noprompt=noprompt),
                                  scenetrans.TwoCloseObjects(p=1.0, noprompt=noprompt),
                                  scenetrans.AObjectAIsCloseToObjectB(p=1.0, within=within, noprompt=noprompt),
                                  scenetrans.ABigObjectAIsCloseToObjectB(p=1.0, within=within, noprompt=noprompt),
                                  scenetrans.ASmallObjectAIsCloseToObjectB(p=1.0, within=within, noprompt=noprompt),
                                  scenetrans.AObjectAIsOnObjectB(p=1.0, within=within, noprompt=noprompt),
                                  scenetrans.AObjectAIsUnderObjectB(p=1.0, within=within, noprompt=noprompt)], 
                                  p=1-alpha_sceneaug)

    compose_list = [
        augmentation.shuffle_points(p=1.0),
        augmentation.rotate_point_cloud(p=0.5),
        augmentation.rotate_point_cloud_z(p=0.5),
        augmentation.jitter_point_cloud(p=0.5),
        scenetrans.OneOf([noaug_trans, aug_trans], p=1.0)
        ]

    train_transforms = T.Compose(compose_list)
    test_transforms = T.Compose([scenetrans.ThisIsAObject(p=1.0, noprompt=noprompt)])
    return train_transforms, test_transforms



class SemanticDataset():
    def __init__(self, **kwargs) -> None:
        self.dataset_eval = kwargs.get('dataset_eval')
        self.train_dataset_dir = kwargs.get('train_dataset_dir')
        self.eval_dataset_dir = kwargs.get('eval_dataset_dir')
        self.num_points = kwargs.get('num_points')
        self.use_normals = kwargs.get('use_normals')
        self.within = kwargs.get("within")
        self.noprompt = kwargs.get("noprompt")
        self.alpha_sceneaug = kwargs.get("alpha_sceneaug")

        self.text_embeddings_path = kwargs.get('text_embeddings_path')
        if self.noprompt:
            zs_classes_text_path = kwargs.get('zs_classes_path')
            gzs_classes_text_path = kwargs.get('gzs_classes_path')
        else:
            zs_classes_text_path = kwargs.get('zs_classes_text_path')
            gzs_classes_text_path = kwargs.get('gzs_classes_text_path')

        self.text_embeddings = sio.loadmat(self.text_embeddings_path)
        self.zs_test_texts = datautil.get_text_lines(zs_classes_text_path)
        self.gzs_test_texts = datautil.get_text_lines(gzs_classes_text_path)

    #------------------------------------------------------------------------------#
    @property
    def train_seen_ds(self):
        train_transforms, _ = get_transforms(alpha_sceneaug=self.alpha_sceneaug, within=self.within, noprompt=self.noprompt)

        if self.dataset_eval == "ModelNet10":
            mapinfo = dict(dsids=ModelNet40_ModelNet10["SEEN_IDS"], 
                           labelmap=[])
        elif self.dataset_eval == "ScanObjectNN":
            mapinfo = dict(dsids=ModelNet40_ScanObjectNN["SEEN_IDS"], 
                           labelmap=[])
        
        return datautil.SemanticModelNet40(text_embeddings_path=self.text_embeddings_path,
                                           dataset_dir=self.train_dataset_dir, 
                                           train=True, 
                                           part="seen",
                                           num_points=self.num_points,
                                           transforms=train_transforms,
                                           mapinfo=mapinfo,
                                           use_normals=self.use_normals)

    #------------------------------------------------------------------------------#
    @property
    def train_seen_noaug_ds(self):
        train_transforms, _ = get_transforms(alpha_sceneaug=0, within=self.within, noprompt=self.noprompt)

        if self.dataset_eval == "ModelNet10":
            mapinfo = dict(dsids=ModelNet40_ModelNet10["SEEN_IDS"], 
                           labelmap=[])
        elif self.dataset_eval == "ScanObjectNN":
            mapinfo = dict(dsids=ModelNet40_ScanObjectNN["SEEN_IDS"], 
                           labelmap=[])

        return datautil.SemanticModelNet40(text_embeddings_path=self.text_embeddings_path,
                                           dataset_dir=self.train_dataset_dir, 
                                           train=True, 
                                           part="seen",
                                           num_points=self.num_points,
                                           transforms=train_transforms,
                                           mapinfo=mapinfo,
                                           use_normals=self.use_normals)

    #------------------------------------------------------------------------------#
    @property
    def test_unseen_noaug_mapped_ds(self):
        _, test_transforms = get_transforms(alpha_sceneaug=0, within=self.within, noprompt=self.noprompt)

        if self.dataset_eval == "ModelNet10":
            mapinfo = dict(dsids=ModelNet40_ModelNet10["ZS_UNSEEN_IDS"], 
                           labelmap=ModelNet40_ModelNet10["ZS_UNSEEN_LABEL_MAP"])
            return datautil.SemanticModelNet40(text_embeddings_path=self.text_embeddings_path,
                                            dataset_dir=self.train_dataset_dir, 
                                            train=False, 
                                            part="unseen",
                                            num_points=self.num_points,
                                            transforms=test_transforms,
                                            mapinfo=mapinfo,
                                            use_normals=self.use_normals)

        elif self.dataset_eval == "ScanObjectNN":
            mapinfo = dict(dsids=ModelNet40_ScanObjectNN["ZS_UNSEEN_IDS"], 
                           labelmap=ModelNet40_ScanObjectNN["ZS_UNSEEN_LABEL_MAP"])
            return datautil.SemanticScanObjectNN(text_embeddings_path=self.text_embeddings_path,
                                                 dataset_dir=self.eval_dataset_dir,
                                                 part="unseen", 
                                                 num_points=self.num_points,
                                                 transforms=test_transforms,
                                                 mapinfo=mapinfo,
                                                 use_normals=self.use_normals)

    #------------------------------------------------------------------------------#
    @property
    def test_seen_noaug_ds(self):
        _, test_transforms = get_transforms(alpha_sceneaug=0, within=self.within, noprompt=self.noprompt)
        if self.dataset_eval == "ModelNet10":
            mapinfo = dict(dsids=ModelNet40_ModelNet10["GZS_SEEN_IDS"], 
                           labelmap=[])
        elif self.dataset_eval == "ScanObjectNN":
            mapinfo = dict(dsids=ModelNet40_ScanObjectNN["GZS_SEEN_IDS"], 
                           labelmap=ModelNet40_ScanObjectNN["GZS_SEEN_LABEL_MAP"])
        return datautil.SemanticModelNet40(text_embeddings_path=self.text_embeddings_path,
                                           dataset_dir=self.train_dataset_dir, 
                                           train=False, 
                                           part="seen",
                                           num_points=self.num_points,
                                           transforms=test_transforms,
                                           mapinfo=mapinfo,
                                           use_normals=self.use_normals)

    #------------------------------------------------------------------------------#
    @property
    def test_unseen_noaug_ds(self):
        _, test_transforms = get_transforms(alpha_sceneaug=0, within=self.within, noprompt=self.noprompt)
        if self.dataset_eval == "ModelNet10":
            mapinfo = dict(dsids=ModelNet40_ModelNet10["GZS_UNSEEN_IDS"], 
                           labelmap=[])
            return datautil.SemanticModelNet40(text_embeddings_path=self.text_embeddings_path,
                                            dataset_dir=self.train_dataset_dir, 
                                            train=False, 
                                            part="unseen",
                                            num_points=self.num_points,
                                            transforms=test_transforms,
                                            mapinfo=mapinfo,
                                            use_normals=self.use_normals)

        elif self.dataset_eval == "ScanObjectNN":
            mapinfo = dict(dsids=ModelNet40_ScanObjectNN["GZS_UNSEEN_IDS"], 
                           labelmap=ModelNet40_ScanObjectNN["GZS_UNSEEN_LABEL_MAP"])
            return datautil.SemanticScanObjectNN(text_embeddings_path=self.text_embeddings_path,
                                                 dataset_dir=self.eval_dataset_dir,
                                                 part="unseen", 
                                                 num_points=self.num_points,
                                                 transforms=test_transforms,
                                                 mapinfo=mapinfo,
                                                 use_normals=self.use_normals)

    #------------------------------------------------------------------------------#
    @property
    def zs_text_embeddings(self):
        return torch.FloatTensor(np.array([self.text_embeddings[txt] for txt in self.zs_test_texts]))

    @property
    def gzs_text_embeddings(self):
        return torch.FloatTensor(np.array([self.text_embeddings[txt] for txt in self.gzs_test_texts]))

