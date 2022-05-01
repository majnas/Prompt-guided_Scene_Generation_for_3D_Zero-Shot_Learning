import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import yaml


class color:
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    BLUE = '\033[1;34;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    RED = '\033[1;31;48m'
    BLACK = '\033[1;30;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'


def save_confusion_matrix(target, pred, class_names, save_path):
    cm = confusion_matrix(y_true=target.reshape(-1,), y_pred=pred.reshape(-1,))        
    score = cm.diagonal()/cm.sum(axis=1)

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=class_names, save_path=save_path)
    return cm, score

def plot_confusion_matrix(cm, classes: list, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, save_path="/tmp/cm.png"):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)


def hm(seen_value: float, unseen_value: float) -> float:
    """
    Calculate Hamrmonic metric for Generalized zero-shot learning
    """
    _hm = 2 * seen_value * unseen_value / (seen_value + unseen_value + 1e-8)
    return _hm


def write_config(config: dict, yml_path: str = "") -> None:
    if yml_path == "":
        yml_path = os.path.join(config["logs_dir"], "config.yml")
    with open(yml_path, 'w') as outfile:
        cfg = config.copy()
        if "device" in cfg: del cfg["device"]
        yaml.dump(cfg, outfile, default_flow_style=False)
