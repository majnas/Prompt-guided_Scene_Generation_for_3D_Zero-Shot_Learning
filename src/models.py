import os

from src import pointnet
from src import pointconv
from src import edgeconv

from src import inductive


def get_classifier(**kwargs):
    backbone = kwargs.get('backbone')
    logs_dir = kwargs.get('logs_dir')
    checkpoints_dir = kwargs.get('checkpoints_dir')
    device = kwargs.get('device')

    print(f"[CLS] >> Store logs in {logs_dir}")
    print(f"[CLS] >> Store checkpoints in {checkpoints_dir}")

    if backbone == "PointNet":
        k = kwargs.get('k',40)  #num_class
        feature_transform = kwargs.get('feature_transform',False)
        return pointnet.PointNetModel(k=k, 
                                      feature_transform=feature_transform, 
                                      logs_dir=logs_dir,
                                      checkpoints_dir=checkpoints_dir,
                                      device=device)

    elif backbone == "PointConv":
        k = kwargs.get('k',40) #num_class
        return pointconv.PointConvModel(k=k,  
                                      logs_dir=logs_dir,
                                      checkpoints_dir=checkpoints_dir,
                                      device=device)

    elif backbone == "EdgeConv":
        k = kwargs.get('k',20) #k for knn
        return edgeconv.EdgeConvModel(k=k,  
                                logs_dir=logs_dir,
                                checkpoints_dir=checkpoints_dir,
                                device=device)                                  
      

def get_inductive(**kwargs):
    backbone = kwargs.get('backbone')
    bachbone_features_size = kwargs.get('bachbone_features_size')
    text_embedding_size = kwargs.get('text_embedding_size')
    embbeding_size = kwargs.get('embbeding_size')
    lr = float(kwargs.get('lr'))
    wd = float(kwargs.get('wd'))
    alpha_sceneaug = kwargs.get('alpha_sceneaug')
    logs_dir = kwargs.get('logs_dir')
    checkpoints_dir = kwargs.get('checkpoints_dir')
    device = kwargs.get('device')
    zs_text_embeddings = kwargs.get('zs_text_embeddings')
    gzs_text_embeddings = kwargs.get('gzs_text_embeddings')
    all_class_names = kwargs.get('all_class_names')
    verbose = kwargs.get('verbose')
    pbar = kwargs.get('pbar')
    epoch = kwargs.get('epoch')

    print(f"[ZSL] >> Store logs in {logs_dir}")
    print(f"[ZSL] >> Store checkpoints in {checkpoints_dir}")

    if backbone == "PointNet":
        k = kwargs.get('k',40)
        feature_transform = kwargs.get('feature_transform',False)
        # pointnet feature extractor
        backbone_features = pointnet.PointNetfeat(global_feat=True, feature_transform=feature_transform)


    elif backbone == "PointConv":
        k = kwargs.get('k',40) #num_class
        backbone_features = pointconv.PointConvfeat(k=k)


    elif backbone == "EdgeConv":
        k = kwargs.get('k',20) #k for knn
        backbone_features = edgeconv.EdgeConvfeat(k=k)



    # create inductive zsl model
    return inductive.InductiveModel(backbone=backbone,
                                    backbone_features=backbone_features,
                                    zs_test_texts=kwargs.get('zs_test_texts'),
                                    gzs_test_texts=kwargs.get('gzs_test_texts'),
                                    zs_text_embeddings=zs_text_embeddings,
                                    gzs_text_embeddings=gzs_text_embeddings,
                                    bachbone_features_size=bachbone_features_size,
                                    text_embedding_size=text_embedding_size,
                                    embbeding_size=embbeding_size,
                                    lr=lr,
                                    wd=wd,
                                    epoch=epoch,
                                    alpha_sceneaug=alpha_sceneaug,
                                    logs_dir=logs_dir,
                                    checkpoints_dir=checkpoints_dir,
                                    all_class_names=all_class_names,
                                    write_summaries=True,
                                    device=device,
                                    verbose=verbose,
                                    pbar=pbar)

