import os
import numpy as np
import scipy.io as sio
import torch
import glob
import h5py
from rich import print
from collections import Counter


def get_text_lines(text_path: str)-> list:
    fh = open(text_path, "r")
    lines = [c.rstrip() for c in fh.readlines()]
    fh.close()
    return lines

# Base code: https://github.com/vinits5/learning3d
def download_modelnet40(dataset_dir: str) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048')):
        url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip' 
        zippath = os.path.join(dataset_dir, "modelnet40_ply_hdf5_2048.zip")
        os.system(f"wget {url} --no-check-certificate -P {dataset_dir}")
        os.system(f"unzip {zippath} -d {dataset_dir}")
        os.system(f'rm {zippath}')

def get_modelnet40_statistics(dataset_dir: str, data: np.ndarray, labels: np.ndarray) -> dict:
    all_class_names = get_text_lines(text_path=os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'))

    statistics = {}
    for label in labels:
        lbl = label[0]
        class_name = all_class_names[lbl] 
        if class_name not in statistics:
            statistics[class_name] = [lbl, 0] # classid, count
        statistics[class_name][1] += 1

    statistics = sorted(statistics.items(), key=lambda x:x[1][0])
    statistics = {k:v[1] for (k,v) in statistics}
    return statistics, all_class_names
    

def load_modelnet40_data(mapinfo: dict, dataset_dir: str, train: bool, part: str, use_normals: bool) -> tuple:
    partition = 'train' if train else 'test'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
        else: data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    # TODO no support for classification
    valid_idxs = [True if i in mapinfo["dsids"] else False for i in all_label]
    all_data, all_label = all_data[valid_idxs], all_label[valid_idxs]
    return all_data, all_label

#########
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40(torch.utils.data.Dataset):
    def __init__(
        self,        
        dataset_dir: str,
        mapinfo: dict,
        train: bool = True,
        part: str = "all",
        num_points: int = 1024,
        download: bool = True,
        randomize_data: bool = False,
        use_normals: bool = False
    ):
        super(ModelNet40, self).__init__()
        if download: download_modelnet40(dataset_dir=dataset_dir)
        self.data, self.labels = load_modelnet40_data(mapinfo=mapinfo,
                                                      dataset_dir=dataset_dir, 
                                                      train=train, 
                                                      part=part, 
                                                      use_normals=use_normals)
        self.statistics, self.all_class_names = get_modelnet40_statistics(dataset_dir=dataset_dir, data=self.data, labels=self.labels)
        self.class_names = list(self.statistics.keys())
        self.num_points = num_points
        self.randomize_data = randomize_data

    def __getitem__(self, idx):
        current_points = self.randomize(idx) if self.randomize_data else self.data[idx].copy()
        current_points[:,0:3]=pc_normalize(current_points[:,0:3])
        current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
    
        return current_points, label

    def __len__(self):
        return self.data.shape[0]

    def randomize(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx, pt_idxs].copy()

    @property
    def n_classes(self):
        return len(self.class_names)

    @property
    def n_samples(self):
        return self.data.shape[0]


class SemanticModelNet40(torch.utils.data.Dataset):
    def __init__(
        self,
        text_embeddings_path: str,
        dataset_dir: str,
        mapinfo: dict,
        train: bool = True,
        part: str = "all",        
        transforms = None, 
        num_points: int = 2048,
        download: bool = True,
        randomize_data: bool = False,
        use_normals: bool = False
    ):
        super(SemanticModelNet40, self).__init__()
        os.makedirs(dataset_dir, exist_ok=True)
        if download: download_modelnet40(dataset_dir=dataset_dir)
        self.data, self.labels = load_modelnet40_data(mapinfo=mapinfo,
                                                      dataset_dir=dataset_dir, 
                                                      train=train, 
                                                      part=part, 
                                                      use_normals=use_normals)
        self.statistics, self.all_class_names = get_modelnet40_statistics(dataset_dir=dataset_dir, data=self.data, labels=self.labels)
        self.class_names = list(self.statistics.keys())
        self.num_points = num_points
        self.randomize_data = randomize_data
        self.part = part
        self.mapinfo = mapinfo

        self.transforms = transforms
        # load text embeddings  
        self.text_embeddings = sio.loadmat(text_embeddings_path)


    def __getitem__(self, idx):
        current_points = self.randomize(idx) if self.randomize_data else self.data[idx].copy()
        current_points = current_points[:self.num_points, :]
        label = self.labels[idx][0]
        item_class_name = self.all_class_names[label]

        sample = {"points": current_points,
                  "text": item_class_name,
                  "label": label,
                  "data": self.data,
                  "labels": self.labels,
                  "all_class_names": self.all_class_names}
        if self.transforms:
            sample = self.transforms(sample)

        label = sample["label"]
        text = sample["text"]

        if self.mapinfo["labelmap"]:
            label = self.mapinfo["labelmap"][label]

        current_points = torch.from_numpy(sample["points"]).float()
        label = torch.tensor(label)
        text_embedding = torch.from_numpy(self.text_embeddings[text]).float()
        # print("text:", text, text_embedding[0, 0:5].cpu().numpy())
        # print("text:", text, "label", label)

        return current_points, label, text_embedding, text

    def __len__(self):
        return self.data.shape[0]

    def randomize(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx, pt_idxs].copy()

    @property
    def n_classes(self):
        return len(self.class_names)

    @property
    def n_samples(self):
        return self.data.shape[0]
      
    @property
    def class_weights(self):
        labelmap = lambda label: label
        if self.map_labels:
            if self.part == "seen":
                labelmap = lambda label : self.mapinfo["labelmap"][label]
        labels_mapped = list(map(labelmap, self.labels.reshape(-1,)))
        return (self.n_samples / (self.n_classes * np.bincount(labels_mapped)))
  













def get_scanobjectnn_statistics(dataset_dir: str, data: np.ndarray, labels: np.ndarray) -> dict:
    all_class_names = get_text_lines(text_path=os.path.join(dataset_dir, 'shape_names.txt'))
    statistics = {}
    for label in labels:
        lbl = label[0]
        class_name = all_class_names[lbl] 
        if class_name not in statistics:
            statistics[class_name] = [lbl, 0] # classid, count
        statistics[class_name][1] += 1

    statistics = sorted(statistics.items(), key=lambda x:x[1][0])
    statistics = {k:v[1] for (k,v) in statistics}
    return statistics, all_class_names


def load_scanobjectnn_test_data(mapinfo: dict, dataset_dir: str, part: str, use_normals: bool) -> tuple:
    all_data = []
    all_label = []
    h5_name = os.path.join(dataset_dir, 'test_objectdataset.h5')
    f = h5py.File(h5_name)
    if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
    else: data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    # TODO no support for classification
    valid_idxs = [True if i in mapinfo["dsids"] else False for i in all_label]
    all_data, all_label = all_data[valid_idxs], all_label[valid_idxs]

    return all_data, all_label.reshape(-1, 1)




class SemanticScanObjectNN(torch.utils.data.Dataset):
    def __init__(
        self,
        text_embeddings_path: str,
        dataset_dir: str,
        mapinfo: dict,
        part: str = "all",        
        transforms = None, 
        num_points: int = 1024,
        download: bool = True,
        randomize_data: bool = False,
        use_normals: bool = False
    ):
        super(SemanticScanObjectNN, self).__init__()
        os.makedirs(dataset_dir, exist_ok=True)
        self.data, self.labels = load_scanobjectnn_test_data(mapinfo=mapinfo, 
                                                             dataset_dir=dataset_dir, 
                                                             part=part, 
                                                             use_normals=use_normals)
        self.statistics, self.all_class_names = get_scanobjectnn_statistics(dataset_dir=dataset_dir, data=self.data, labels=self.labels)
        self.class_names = list(self.statistics.keys())
        self.num_points = num_points
        self.randomize_data = randomize_data
        self.part = part
        self.mapinfo = mapinfo

        self.transforms = transforms
        # load text embeddings  
        self.text_embeddings = sio.loadmat(text_embeddings_path)


    def __getitem__(self, idx):
        current_points = self.randomize(idx) if self.randomize_data else self.data[idx].copy()
        current_points = current_points[:self.num_points, :]
        label = self.labels[idx][0]
        item_class_name = self.all_class_names[label]

        sample = {"points": current_points,
                  "text": item_class_name,
                  "label": label,
                  "data": self.data,
                  "labels": self.labels,
                  "all_class_names": self.all_class_names}
        if self.transforms:
            sample = self.transforms(sample)

        label = sample["label"]
        text = sample["text"]

        if self.mapinfo["labelmap"]:
            label = self.mapinfo["labelmap"][label]

        current_points = torch.from_numpy(sample["points"]).float()
        label = torch.tensor(label)
        # print("text:", text, "label", label)
        text_embedding = torch.from_numpy(self.text_embeddings[text]).float()
        # print("text:", text, text_embedding[0, 0:5].cpu().numpy())

        return current_points, label, text_embedding, text


    def __len__(self):
        return self.data.shape[0]

    @property
    def n_classes(self):
        return len(self.class_names)

    @property
    def n_samples(self):
        return self.data.shape[0]






# def test_ModelNet40_ModelNet10():
#     text_embeddings_path = "./data/ModelNet/bert_txt_embedding_modelnet40.mat"
#     dataset_dir = "./data/ModelNet/"
#     dataset_train_eval = "ModelNet40_ModelNet10"
#     train_transforms = T.Compose([scenetrans.AObjectAIsCloseToObjectB(p=1.0, within=False)])
#     test_transforms = T.Compose([scenetrans.ThisIsAObject(p=1.0)])

#     train_ds = SemanticModelNet40(dataset_train_eval=dataset_train_eval,
#                                   text_embeddings_path=text_embeddings_path,
#                                   dataset_dir=dataset_dir, 
#                                   transforms=train_transforms,
#                                   train=True, 
#                                   part="seen",
#                                   map_labels=True)
#     test_ds = SemanticModelNet40(dataset_train_eval=dataset_train_eval,
#                                  text_embeddings_path=text_embeddings_path,
#                                  dataset_dir=dataset_dir, 
#                                  transforms=test_transforms,
#                                  train=False, 
#                                  part="seen",
#                                  map_labels=True)

#     train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#     test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

#     # test dataset shapes
#     #-----------------------------------------------------------------------------#
#     print(f"[ZSL] >> Train Samples -------> {train_ds.n_samples}")
#     print(f"[ZSL] >> Test Samples --------> {test_ds.n_samples}")
#     print(f"[ZSL] >> n_classes -----------> {train_ds.n_classes}")
#     print(train_ds.statistics)

#     current_points, label, text_embedding, text = next(iter(train_dl))
#     print(f"[ZSL] >> current_points ------> {current_points.shape}")
#     print(f"[ZSL] >> label ---------------> {label.shape}")
#     print(f"[ZSL] >> text_embedding ------> {text_embedding.shape}")
#     print(text)

    # extract class weights
    #-----------------------------------------------------------------------------#
    # batch_lables_all = []
    # for batch_points, batch_lables, batch_text_embedding, batch_texts in train_dl:
    #     batch_lables_all.append(batch_lables.view(-1,).numpy())

    # batch_lables_all = np.concatenate(batch_lables_all, 0)
    # print(Counter(batch_lables_all))
    # print("class_weights", train_ds.class_weights)
    #-----------------------------------------------------------------------------#



def test_ModelNet40_ScanObjectNN():
    text_embeddings_path = "./data/ModelNet/bert_txt_embedding_modelnet40.mat"
    train_dataset_dir = "./data/ModelNet/"
    test_dataset_dir = "./data/ScanObjectNN/"
    dataset_train_eval = "ModelNet40_ScanObjectNN"
    train_transforms = T.Compose([scenetrans.AObjectAIsCloseToObjectB(p=1.0, within=False)])
    test_transforms = T.Compose([scenetrans.ThisIsAObject(p=1.0)])

    # train_ds = SemanticModelNet40(dataset_train_eval=dataset_train_eval,
    #                               text_embeddings_path=text_embeddings_path,
    #                               dataset_dir=train_dataset_dir, 
    #                               transforms=train_transforms,
    #                               train=True, 
    #                               part="seen",
    #                               map_labels=True)
    test_ds = SemanticScanObjectNN(text_embeddings_path=text_embeddings_path,
                                   dataset_dir=test_dataset_dir, 
                                   transforms=test_transforms,
                                   part="seen",
                                   map_labels=True)

    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    # test dataset shapes
    #-----------------------------------------------------------------------------#
    # print(f"[ZSL] >> Train Samples -------> {train_ds.n_samples}")
    print(f"[ZSL] >> Test Samples --------> {test_ds.n_samples}")
    print(f"[ZSL] >> n_classes -----------> {test_ds.n_classes}")
    print(test_ds.statistics)

    # current_points, label, text_embedding, text = next(iter(train_dl))
    # print(f"[ZSL] >> current_points ------> {current_points.shape}")
    # print(f"[ZSL] >> label ---------------> {label.shape}")
    # print(f"[ZSL] >> text_embedding ------> {text_embedding.shape}")
    # print(text)

    # extract class weights
    #-----------------------------------------------------------------------------#
    # batch_lables_all = []
    # for batch_points, batch_lables, batch_text_embedding, batch_texts in train_dl:
    #     batch_lables_all.append(batch_lables.view(-1,).numpy())

    # batch_lables_all = np.concatenate(batch_lables_all, 0)
    # print(Counter(batch_lables_all))
    # print("class_weights", train_ds.class_weights)
    #-----------------------------------------------------------------------------#



if __name__ == "__main__":
    import torchvision.transforms as T
    import scenetrans

    # test_ModelNet40_ModelNet10()
    test_ModelNet40_ScanObjectNN()


    