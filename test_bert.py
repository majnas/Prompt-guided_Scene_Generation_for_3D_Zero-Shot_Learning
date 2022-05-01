import scipy.io as sio
import numpy as np
from src import dataset
from src import datautil
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

config = {"text_embeddings_path": "./data/ModelNet/bert_txt_embedding_modelnet40.mat",
          "zs_classes_text_path": "./data/ModelNet/zs_classes_modelnet40.txt",
          "gzs_classes_text_path": "./data/ModelNet/gzs_classes_modelnet40.txt",
          }
# zs_classes_embeddings, gzs_classes_embeddings = dataset.get_test_text_embeddings(**config)
# zs_test_classes = datautil.get_text_lines(config["zs_classes_text_path"])
# gzs_test_classes = datautil.get_text_lines(config["gzs_classes_text_path"])
# print("zs_test_classes", zs_test_classes)
# print("zs_classes_embeddings", zs_classes_embeddings[:,:,0:5])


# config = {"text_embeddings_path": "./data/ModelNet/bert_txt_embedding_modelnet40.mat",
#           "zs_classes_text_path": "./data/ModelNet/zs_text_classes_modelnet40.txt",
#           "gzs_classes_text_path": "./data/ModelNet/gzs_text_classes_modelnet40.txt",
#           }
# zs_text_embeddings, gzs_text_embeddings = dataset.get_test_text_embeddings(**config)
# zs_test_texts = datautil.get_text_lines(config["zs_classes_text_path"])
# gzs_test_texts = datautil.get_text_lines(config["gzs_classes_text_path"])


# zs_embeddings = torch.vstack((zs_classes_embeddings.squeeze(), zs_text_embeddings.squeeze()))
# gzs_embeddings = torch.vstack((gzs_classes_embeddings.squeeze(), gzs_text_embeddings.squeeze()))
# print("zs_embeddings", zs_embeddings.shape)
# print("gzs_embeddings", gzs_embeddings.shape)

# writer.add_embedding(zs_embeddings.squeeze(), metadata=zs_test_classes + zs_test_texts, tag="ZS")
# writer.add_embedding(gzs_embeddings.squeeze(), metadata=gzs_test_classes + gzs_test_texts, tag="GZS")


# texts = ["chair", "This is a chair."]
# text_embeddings = sio.loadmat(config["text_embeddings_path"])
# embeddings = [text_embeddings[t] for t in texts]
# embeddings = np.vstack(embeddings)
# writer.add_embedding(embeddings, metadata=texts, tag="C1")


texts = ["chair", "This is a chair.", "This is a chair.", "sofa", "This is a sofa.", "A chair is close to sofa.", "A sofa is close to chair."]
text_embeddings = sio.loadmat(config["text_embeddings_path"])
embeddings = [text_embeddings[t] for t in texts]
embeddings = np.vstack(embeddings)
writer.add_embedding(embeddings, metadata=texts, tag="C2")

writer.flush()
