import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import glob
from time import time
from tqdm import tqdm
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class EdgeConvfeat(nn.Module):
    def __init__(self, k=20, dropout=0.5, emb_dims=1024,output_channels=40):
        super(EdgeConvfeat, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x, None, None 

class DGCNN(nn.Module):
    def __init__(self, k=20, dropout=0.5, emb_dims=1024,output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class EdgeConvModel():
    def __init__(self, 
                 k: int =20,
                 logs_dir: str = "logs/edgeconv_modelnet",
                 checkpoints_dir: str = 'checkpoints/edgeconv_modelnet',
                 device: str = "cpu") -> None:
        self.classifier = DGCNN(k=k)

        # if opt.model != '':
        #     classifier.load_state_dict(torch.load(opt.model))

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.scheduler = CosineAnnealingLR(self.optimizer, 2, eta_min=0.001)
        self.classifier = nn.DataParallel(self.classifier)
        self.classifier.to(device)
        self.criterion=cal_loss
        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir
        self.test_acc_best = 0
        self.device = device       
        self.writer = SummaryWriter()

        # create directory
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train(self, train_dl, test_dl, n_epochs: int):
        for epoch in tqdm(range(n_epochs)):
            # train one epoch
            train_loss_val, train_acc_val = self.train_one_epoch(train_dl=train_dl)
            self.writer.add_scalar("Train/loss", train_loss_val, epoch)
            self.writer.add_scalar("Train/acc", train_acc_val, epoch)

            # evaluate model
            test_loss_val, test_acc_val = self.test(test_dl=test_dl)
            self.writer.add_scalar("Test/loss", test_loss_val, epoch)
            self.writer.add_scalar("Test/acc", test_acc_val, epoch)

            # step learning rate
            self.scheduler.step()

            # save model
            save_best = test_acc_val >= self.test_acc_best
            if save_best: self.test_acc_best = test_acc_val
            self.save(epoch=epoch, save_best=save_best)

        self.writer.flush()


    def train_one_epoch(self, train_dl):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(train_dl, leave=False)
        for i, (batch_points, batch_lables) in enumerate(pbar):
            batch_lables = batch_lables[:, 0]
            batch_points = batch_points.transpose(2, 1)
            batch_points=batch_points[:,:3,:]
            # print(batch_points.shape)
            # batch_points=batch_points
            batch_points, batch_lables = batch_points.to(self.device), batch_lables.to(self.device)
            self.optimizer.zero_grad()
            classifier = self.classifier.train()
            
            pred= classifier(batch_points)
            loss = self.criterion(pred, batch_lables)
            loss.backward()
            self.optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(batch_lables.data).cpu().sum()

            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += batch_points.size()[0]
            mean_loss = round(total_loss / float(total_samples),4)
            mean_acc = round(total_correct / float(total_samples),4)
            pbar.set_description(f"best_test_acc:{self.test_acc_best:<.4f} train_mloss:{mean_loss:<.4f} train_macc:{mean_acc:<.4f}")

        return mean_loss, mean_acc

    @torch.no_grad()
    def test(self, test_dl):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(test_dl, leave=False)
        classifier = self.classifier.eval()
        for i, (batch_points, batch_lables) in enumerate(pbar):
            batch_lables = batch_lables[:, 0]
            batch_points = batch_points.transpose(2, 1)
            batch_points = batch_points[:,:3,:]
            batch_points, batch_lables = batch_points.to(self.device), batch_lables.to(self.device)
            pred= classifier(batch_points)
            loss = self.criterion(pred, batch_lables)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(batch_lables.data).cpu().sum()

            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += batch_points.size()[0]
            mean_loss = round(total_loss / float(total_samples),4)
            mean_acc = round(total_correct / float(total_samples), 4)
            pbar.set_description(f"best_test_acc:{self.test_acc_best:<.4f} test_mloss:{mean_loss:<.4f} test_macc:{mean_acc:<.4f}")
        return mean_loss, mean_acc


    def save(self, epoch : int, save_best: bool = True) -> None:
        torch.save(self.classifier.state_dict(), f"{self.checkpoints_dir}/edgeconv_model_{epoch}_{self.test_acc_best}.pth")
        if save_best:
            # remove all previous best models
            [os.remove(f) for f in glob.glob(os.path.join(self.checkpoints_dir, "*best*"))]
            torch.save(self.classifier.state_dict(), f"{self.checkpoints_dir}/edgeconv_model_best_{self.test_acc_best}.pth")

