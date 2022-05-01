from __future__ import print_function
import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from src import visualization

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss



class PointNetModel():
    def __init__(self, 
                 k: int = 40, 
                 feature_transform: bool = False,
                 logs_dir: str = "logs/PointNet",
                 checkpoints_dir: str = 'checkpoints/PointNet',
                 device: str = "cpu") -> None:
        self.classifier = PointNetCls(k=k, feature_transform=feature_transform)

        # if opt.model != '':
        #     classifier.load_state_dict(torch.load(opt.model))

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.classifier = nn.DataParallel(self.classifier)
        self.classifier.to(device)
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
            
            # batch_points_np = batch_points.numpy()
            # batch_lables_np = batch_lables.numpy()
            # for points, label in zip(batch_points_np, batch_lables_np):
            #     obj_path = os.path.join(self.logs_dir, f"object_{label[0]}.obj")
            #     visualization.export_color_point_cloud(points=points, obj_path=obj_path)

            batch_lables = batch_lables[:, 0]
            batch_points = batch_points.transpose(2, 1)
            batch_points=batch_points[:,:3,:]
            batch_points, batch_lables = batch_points.to(self.device), batch_lables.to(self.device)
            self.optimizer.zero_grad()
            classifier = self.classifier.train()
            pred, trans, trans_feat = classifier(batch_points)
            loss = F.nll_loss(pred, batch_lables)
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
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
        with torch.no_grad():
            for i, (batch_points, batch_lables) in enumerate(pbar):
                batch_lables = batch_lables[:, 0]
                batch_points = batch_points.transpose(2, 1)
                batch_points=batch_points[:,:3,:]
                batch_points, batch_lables = batch_points.to(self.device), batch_lables.to(self.device)
                pred, _, _ = classifier(batch_points)
                loss = F.nll_loss(pred, batch_lables)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(batch_lables.data).cpu().sum()

                total_loss += loss.item()
                total_correct += correct.item()
                total_samples += batch_points.size()[0]
                mean_loss = round(total_loss / float(total_samples), 4)
                mean_acc = round(total_correct / float(total_samples), 4)
                pbar.set_description(f"best_test_acc:{self.test_acc_best:<.4f} test_mloss:{mean_loss:<.4f} test_macc:{mean_acc:<.4f}")
        return mean_loss, mean_acc


    def save(self, epoch : int, save_best: bool = True) -> None:
        torch.save(self.classifier.state_dict(), f"{self.checkpoints_dir}/pointnet_cls_model_{epoch}_{self.test_acc_best}.pth")
        if save_best:
            # remove all previous best models
            [os.remove(f) for f in glob.glob(os.path.join(self.checkpoints_dir, "*best*"))]
            torch.save(self.classifier.state_dict(), f"{self.checkpoints_dir}/pointnet_cls_model_best_{self.test_acc_best}.pth")






if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())


