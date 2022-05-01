import os
import glob
from tqdm import tqdm, utils
import torch
# from torch import Tensor
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any
from src import util
from src.util import color as clr
import numpy as np


class ZSLoss(nn.Module):
    def __init__(self, device, t : float = 0.1):
        super(ZSLoss, self).__init__()
        self.device = device
        self.t = nn.Parameter(torch.ones([])* np.log(1/t)).exp().to(device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feature_embeddings, semantic_embeddings):
        feature_embeddings = F.normalize(feature_embeddings)
        semantic_embeddings = F.normalize(semantic_embeddings)

        # scaled pairwise cosine similarities [n, n]
        logits = feature_embeddings @ semantic_embeddings.t() * self.t

        # symmetric loss function
        batch_size = feature_embeddings.shape[0]
        labels = torch.arange(batch_size).to(self.device)

        loss_features = self.loss(input=logits, target=labels)
        loss_semantics = self.loss(input=logits.T, target=labels)
        loss = (loss_features + loss_semantics) / 2
        # print("loss_it", loss_features, loss_semantics)

        return loss

class ZSTestLoss(nn.Module):
    """
        Calculate input bach data loss respect to their label
    """
    def __init__(self, device, t : float = 0.1):
        super(ZSTestLoss, self).__init__()
        self.device = device
        self.t = nn.Parameter(torch.ones([])* np.log(1/t)).exp().to(device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feature_embeddings: torch.Tensor, 
                      semantic_embeddings: torch.Tensor, 
                      labels: torch.Tensor):
        feature_embeddings = F.normalize(feature_embeddings)
        semantic_embeddings = F.normalize(semantic_embeddings)
        logits = feature_embeddings @ semantic_embeddings.t() * self.t
        loss = self.loss(input=logits, target=labels)

        return loss


class InductiveBaseModel(nn.Module):
    def __init__(self, 
                 bachbone_features: torch.nn.Module,
                 bachbone_features_size: int = 1024,
                 text_embedding_size: int = 768,
                 embbeding_size: int = 128):
        super(InductiveBaseModel, self).__init__()
        self.bachbone_features = bachbone_features
        self.bachbone_features_size = bachbone_features_size
        self.text_embedding_size = text_embedding_size
        self.embbeding_size = embbeding_size
        
        self.fefc1 = nn.Linear(bachbone_features_size, 512)
        self.fefc2 = nn.Linear(512, embbeding_size)

        self.sefc1 = nn.Linear(text_embedding_size,1024)
        self.sefc2 = nn.Linear(1024, 512)
        self.sefc3 = nn.Linear(512, embbeding_size)

    def forward(self, x, e):
        # split points of normals are included
        x = (x,) if x.shape[1] == 3 else (x[:,:3,:], x[:,3:6,:])
        bf, _, _ = self.bachbone_features(*x)
        bf = bf.view(-1, self.bachbone_features_size)

        # backbone features mapped to reduced embedding space
        fe = self.fefc2(F.relu(self.fefc1(bf)))

        # input embedding mapped to reduced embedding space
        e = e.view(-1, self.text_embedding_size)
        se = self.sefc3(F.relu(self.sefc2(F.relu(self.sefc1(e)))))

        return fe, se



class InductiveModel():
    def __init__(self,
                 backbone: str,
                 backbone_features: torch.nn.Module, 
                 zs_test_texts: list,
                 gzs_test_texts: list,
                 zs_text_embeddings: torch.Tensor,
                 gzs_text_embeddings: torch.Tensor,
                 bachbone_features_size: int = 1024,
                 text_embedding_size: int = 768,
                 embbeding_size: int = 128,
                 lr: float = 1e-3,
                 wd: float = 1e-4,
                 epoch: int = 100,
                 alpha_sceneaug: float = 0.1, # scene_augmentation_regulizer
                 logs_dir: str = "logs/inductive_pointnet",
                 checkpoints_dir: str = 'checkpoints/inductive_pointnet',
                 all_class_names: list = [],
                 device: str = "cpu",
                 write_summaries: bool = True,
                 verbose: bool = True,
                 pbar: bool = True) -> None:

        self.model = InductiveBaseModel(bachbone_features=backbone_features,
                                        bachbone_features_size=bachbone_features_size,
                                        text_embedding_size=text_embedding_size,
                                        embbeding_size=embbeding_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
        if backbone == "PointNet":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        elif backbone == "PointConv":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        elif backbone == "EdgeConv":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epoch, eta_min=lr) 

        self.zsloss_noaug = ZSLoss(device=device, t=0.07)
        self.zsloss = ZSLoss(device=device, t=0.07)
        self.zstestloss = ZSTestLoss(device=device, t=0.07)
        self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.zs_test_texts = zs_test_texts
        self.gzs_test_texts = gzs_test_texts
        self.zs_text_embeddings = zs_text_embeddings.to(self.device)
        self.gzs_text_embeddings = gzs_text_embeddings.to(self.device)
        self.all_class_names = all_class_names
        self.alpha_sceneaug = alpha_sceneaug
        self.write_summaries = write_summaries
        self.verbose = verbose
        self.pbar = pbar

        # create directory
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if write_summaries:
            self.writer = SummaryWriter(log_dir=self.logs_dir)

        # init parameters
        self.zs_test_acc_best = 0 
        self.gzs_test_acc_best = [0, 0, 0] # [sacc, uacc, hmacc]
        self.t_ret = None
        self.tgzss_ret = None
        self.zs_ret = None
        self.gzss_ret = None
        self.gzsu_ret = None
        self.score = []
        self.zs_score_best = []

        # write zs_text_embeddings and gzs_text_embeddings
        if self.write_summaries:
            self.write_text_embeddings()


    def train(self, train_seen_dl: torch.utils.data.DataLoader, 
                    test_seen_dl: torch.utils.data.DataLoader, 
                    test_unseen_mapped_dl: torch.utils.data.DataLoader, 
                    test_unseen_dl: torch.utils.data.DataLoader, 
                    n_epochs: int):
        
        pbar = tqdm(range(n_epochs)) if self.verbose else range(n_epochs)
        pbar = pbar if self.pbar else range(n_epochs)
        for epoch in pbar:
            # train one epoch
            self.t_ret = self.train_one_epoch(train_seen_dl=train_seen_dl)

            # evaluate model performance on train seen part according to gzs_text_embeddings
            # self.tgzss_ret = self.test(test_dl=train_seen_noaug_dl, 
            #                            test_text_embeddings=self.gzs_text_embeddings)

            # evaluate ZSL model
            self.zs_ret = self.test(test_dl=test_unseen_mapped_dl, 
                                    test_text_embeddings=self.zs_text_embeddings)

            # evaluate GZSL model on seen part
            self.gzss_ret = self.test(test_dl=test_seen_dl, 
                                      test_text_embeddings=self.gzs_text_embeddings)

            # evaluate GZSL model on unseen part
            self.gzsu_ret = self.test(test_dl=test_unseen_dl, 
                                      test_text_embeddings=self.gzs_text_embeddings)

            # step learning rate
            self.scheduler.step()

            # write summaries
            if self.write_summaries:
                self.write_all_summaries(epoch=epoch)


            # save confusion matrix
            # self.save_confusion_matrix(epoch=epoch)
            # print("zs_score_best", self.zs_score_best)

            # save model
            self.save(epoch=epoch)

            # desc = f"epoch={clr.RED}{epoch}{clr.END} best_zs_test_acc:{clr.RED}{self.zs_test_acc_best:<.4f}{clr.END}" + \
            #        f" best_gzs_test_acc:{clr.RED}({self.gzs_test_acc_best[0]:<.4f}, " + \
            #        f"{self.gzs_test_acc_best[1]:<.4f}, {self.gzs_test_acc_best[2]:<.4f}){clr.END}"
            # print(desc)

        if self.write_summaries:
            self.writer.flush()


    def train_one_epoch(self, train_seen_dl: torch.utils.data.DataLoader):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        self.model = self.model.train()
        iters = len(train_seen_dl.dataset) // train_seen_dl.batch_size
        pbar = tqdm(train_seen_dl, total=iters, leave=False) if self.verbose else train_seen_dl
        pbar = pbar if self.pbar else train_seen_dl
        for i, batch in enumerate(pbar):
            # not augmented batch
            batch_seen_points, batch_label, batch_seen_semantic, batch_text = batch
            batch_seen_points = batch_seen_points.transpose(2, 1)

            batch_seen_points, batch_seen_semantic = batch_seen_points.to(self.device), batch_seen_semantic.to(self.device)
            feature_embeddings, semantic_embeddings = self.model(batch_seen_points, batch_seen_semantic)
            loss = self.zsloss(feature_embeddings=feature_embeddings, semantic_embeddings=semantic_embeddings)

            # gather losses
            loss_totall = loss
            self.optimizer.zero_grad()
            loss_totall.backward(retain_graph=True)
            self.optimizer.step()

            similarity = F.normalize(feature_embeddings) @ F.normalize(semantic_embeddings).t()
            preds = torch.argmax(similarity, dim=1)
            target = torch.arange(batch_seen_points.shape[0]).to(self.device)
            correct = torch.sum(preds == target)

            # scene augmented
            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += batch_seen_points.size()[0]

            mean_loss = total_loss / float(total_samples)
            mean_acc = total_correct / float(total_samples) * 100

            if self.verbose:
                desc = f"best_zs_test_acc:{clr.RED}{self.zs_test_acc_best:<.2f}{clr.END}" + \
                       f" best_gzs_test_acc:{clr.RED}({self.gzs_test_acc_best[0]:<.2f}, {self.gzs_test_acc_best[1]:<.2f}, {self.gzs_test_acc_best[2]:<.2f}){clr.END}" + \
                       f" train_abloss:{clr.BLUE}{mean_loss:<.4f}{clr.END}" + \
                       f" train_abacc:{clr.BLUE}{mean_acc:<.2f}{clr.END}"
                if self.pbar:
                    pbar.set_description(desc=desc)
                elif (i%100==0):
                    print(f"iter:{i:4} " + desc)

        return mean_loss, mean_acc


    @torch.no_grad()
    def test(self, test_dl: torch.utils.data.DataLoader, test_text_embeddings):
        total_loss = 0
        total_batch_loss = 0
        total_correct = 0
        total_batch_correct = 0
        total_samples = 0
        all_labels = torch.empty([1, 0])
        all_preds = torch.empty([1, 0])
        all_feature_embeddings = []
        pbar = tqdm(test_dl, leave=False) if self.verbose else test_dl
        pbar = pbar if self.pbar else test_dl
        self.model = self.model.eval()
        for i, (batch_points, batch_label, batch_semantic, batch_text) in enumerate(pbar):
            all_labels = torch.cat((all_labels, batch_label.view(1,-1)), dim=1)
            batch_points = batch_points.transpose(2, 1)
            batch_points, batch_label, batch_semantic = batch_points.to(self.device), batch_label.to(self.device), batch_semantic.to(self.device)

            # evaluate on batch text embeddings
            batch_feature_embeddings, batch_semantic_embeddings = self.model(batch_points, batch_semantic)
            all_feature_embeddings.append(batch_feature_embeddings)
            batch_loss = self.zsloss(feature_embeddings=batch_feature_embeddings, semantic_embeddings=batch_semantic_embeddings)
            batch_similarity = F.normalize(batch_feature_embeddings) @ F.normalize(batch_semantic_embeddings).t()
            batch_preds = torch.argmax(batch_similarity, dim=1)
            target = torch.arange(batch_points.size()[0]).to(self.device)
            batch_correct = torch.sum(batch_preds == target)

            # evaluate on zero shot test text embeddings
            feature_embeddings, semantic_embeddings = self.model(batch_points, test_text_embeddings)
            loss = self.zstestloss(feature_embeddings=feature_embeddings, 
                                   semantic_embeddings=semantic_embeddings, 
                                   labels=batch_label)
            similarity = F.normalize(feature_embeddings) @ F.normalize(semantic_embeddings).t()
            preds = torch.argmax(similarity, dim=1)
            correct = torch.sum(preds == batch_label)

            all_preds = torch.cat((all_preds, preds.cpu().view(1,-1)), dim=1)

            total_samples += batch_points.size()[0]
            total_batch_loss += batch_loss.item()
            total_batch_correct += batch_correct.item()
            mean_batch_loss = total_batch_loss / float(total_samples)
            mean_batch_acc = total_batch_correct / float(total_samples)

            total_loss += loss.item()
            total_correct += correct.item()
            mean_loss = total_loss / float(total_samples)
            mean_acc = total_correct / float(total_samples)

            if self.verbose:
                desc = f"best_zs_test_acc:{clr.RED}{self.zs_test_acc_best:<.2f}{clr.END}" + \
                       f" best_gzs_test_acc:{clr.RED}({self.gzs_test_acc_best[0]:<.2f}, {self.gzs_test_acc_best[1]:<.2f}, {self.gzs_test_acc_best[2]:<.2f}){clr.END}" + \
                       f" test_loss:{clr.BLUE}{mean_loss:<.4f}{clr.END}" + \
                       f" test_acc:{clr.BLUE}{mean_acc:<.2f}{clr.END}" + \
                       f" test_bloss:{clr.BLUE}{mean_batch_loss:<.4f}{clr.END}" + \
                       f" test_bacc:{clr.BLUE}{mean_batch_acc:<.2f}{clr.END}"
                if self.pbar:
                    pbar.set_description(desc=desc)
                elif (i%100==0):
                    print(f"iter:{i:4} " + desc)

        return mean_loss, mean_acc, mean_batch_loss, mean_batch_acc, (all_preds, all_labels, all_feature_embeddings, semantic_embeddings)


    def write_text_embeddings(self):
        # print("self.gzs_text_embeddings.squeeze()", self.gzs_text_embeddings.squeeze().shape)
        # FIXME
        # self.writer.add_embedding(self.gzs_text_embeddings.squeeze(), 
        #                           metadata=self.all_class_names, 
        #                           tag="Centers")
        pass


    def write_all_summaries(self, epoch: int) -> None:        
        if self.t_ret:
            train_loss_val, train_acc_val = self.t_ret
            self.writer.add_scalar("Train/abloss", train_loss_val, epoch)
            self.writer.add_scalar("Train/abacc", train_acc_val, epoch)

        if self.tgzss_ret:
            gzss_train_loss_val, gzss_train_acc_val, gzss_train_batch_loss_val, gzss_train_batch_acc_val, tgzss_params = self.tgzss_ret
            _, tgzss_all_labels, tgzss_all_feature_embeddings, tgzss_semantic_embeddings = tgzss_params
            self.writer.add_scalar("Train_GZS/sloss", gzss_train_loss_val, epoch)
            self.writer.add_scalar("Train_GZS/sacc", gzss_train_acc_val, epoch)
            self.writer.add_scalar("Train_GZS/sbloss", gzss_train_batch_loss_val, epoch)
            self.writer.add_scalar("Train_GZS/sbacc", gzss_train_batch_acc_val, epoch)

        if self.zs_ret:
            zs_test_loss_val, zs_test_acc_val, zs_test_batch_loss_val, zs_test_batch_acc_val, zs_params = self.zs_ret
            _, zs_all_labels, zs_all_feature_embeddings, zs_semantic_embeddings = zs_params
            self.writer.add_scalar("Test_ZS/loss", zs_test_loss_val, epoch)
            self.writer.add_scalar("Test_ZS/acc", zs_test_acc_val, epoch)
            self.writer.add_scalar("Test_ZS/bloss", zs_test_batch_loss_val, epoch)
            self.writer.add_scalar("Test_ZS/bacc", zs_test_batch_acc_val, epoch)
            # zs_all_feature_embeddings = torch.vstack(zs_all_feature_embeddings)
            # zs_all_class_names = [self.all_class_names[int(i)] for i in zs_all_labels.view(-1,).cpu().numpy()]
            # self.writer.add_embedding(zs_all_feature_embeddings, metadata=zs_all_class_names, tag="ZS", global_step=epoch)

        if self.gzss_ret:
            gzss_test_loss_val, gzss_test_acc_val, gzss_test_batch_loss_val, gzss_test_batch_acc_val, gzss_params = self.gzss_ret
            _, gzss_all_labels, gzss_all_feature_embeddings, gzss_semantic_embeddings = gzss_params
            self.writer.add_scalar("Test_GZS/sloss", gzss_test_loss_val, epoch)
            self.writer.add_scalar("Test_GZS/sacc", gzss_test_acc_val, epoch)
            self.writer.add_scalar("Test_GZS/sbloss", gzss_test_batch_loss_val, epoch)
            self.writer.add_scalar("Test_GZS/sbacc", gzss_test_batch_acc_val, epoch)
            # gzss_all_feature_embeddings = torch.vstack(gzss_all_feature_embeddings)
            # print("gzss_all_feature_embeddings",gzss_all_feature_embeddings.shape)
            # gzss_all_class_names = [self.all_class_names[int(i)] for i in gzss_all_labels.view(-1,).cpu().numpy()]
            # self.writer.add_embedding(gzss_all_feature_embeddings, metadata=gzss_all_class_names, tag="GZSS", global_step=epoch)

        if self.gzsu_ret:
            gzsu_test_loss_val, gzsu_test_acc_val, gzsu_test_batch_loss_val, gzsu_test_batch_acc_val, gzsu_params = self.gzsu_ret
            _, gzsu_all_labels, gzsu_all_feature_embeddings, gzsu_semantic_embeddings = gzsu_params
            self.writer.add_scalar("Test_GZS/uloss", gzsu_test_loss_val, epoch)
            self.writer.add_scalar("Test_GZS/uacc", gzsu_test_acc_val, epoch)
            self.writer.add_scalar("Test_GZS/ubloss", gzsu_test_batch_loss_val, epoch)
            self.writer.add_scalar("Test_GZS/ubacc", gzsu_test_batch_acc_val, epoch)
            self.gzs_test_texts = [s.replace("This is a", "").replace(".", "").strip() for s in self.gzs_test_texts]
            self.writer.add_embedding(gzsu_semantic_embeddings, metadata=self.gzs_test_texts, tag="GZSU", global_step=epoch)

        if self.gzss_ret and self.gzsu_ret:       
            gzs_all_feature_embeddings = torch.vstack(gzss_all_feature_embeddings + gzsu_all_feature_embeddings)
            gzs_all_labels = torch.hstack((gzss_all_labels, gzsu_all_labels))
            gzs_all_class_names = [["seen", self.all_class_names[int(i)]] for i in gzss_all_labels.view(-1,).cpu().numpy()] + \
                                   [["unseen", self.all_class_names[int(i)]] for i in gzsu_all_labels.view(-1,).cpu().numpy()]
            self.writer.add_embedding(gzs_all_feature_embeddings, metadata=gzs_all_class_names, metadata_header=["seen_unseen","class_name"], tag="GZS", global_step=epoch)

            self.writer.add_scalar("Test_GZS/hmloss", util.hm(gzss_test_loss_val, gzsu_test_loss_val), epoch)
            self.writer.add_scalar("Test_GZS/hmacc", util.hm(gzss_test_acc_val, gzsu_test_acc_val), epoch)
            self.writer.add_scalar("Test_GZS/hmbloss", util.hm(gzss_test_batch_loss_val, gzsu_test_batch_loss_val), epoch)
            self.writer.add_scalar("Test_GZS/hmbacc", util.hm(gzss_test_batch_acc_val, gzsu_test_batch_acc_val), epoch)


    def save_confusion_matrix(self, epoch : int) -> None:
        # _, _, _, _, gzss_params = self.gzss_ret
        # gzss_all_preds, gzss_all_labels, _, _ = gzss_params

        # _, _, _, _, gzsu_params = self.gzsu_ret
        # gzsu_all_preds, gzsu_all_labels, _, _ = gzsu_params

        # print(gzss_all_preds.shape, gzss_all_labels.shape)
        # print(gzsu_all_preds.shape, gzsu_all_labels.shape)

        # all_labels = np.hstack((gzss_all_labels.cpu().numpy(), gzsu_all_labels.cpu().numpy()))
        # all_preds = np.hstack((gzss_all_preds.cpu().numpy(), gzsu_all_preds.cpu().numpy()))

        # util.save_confusion_matrix(target=all_labels, 
        #                             pred=all_preds, 
        #                             class_names=[f"class{i}" for i in range(len(np.unique(all_labels)))],
        #                             save_path=os.path.join(self.logs_dir, f"cm_{epoch}.png"))

        _, _, _, _, zs_params = self.zs_ret
        zs_all_preds, zs_all_labels, _, _ = zs_params

        zs_all_preds = zs_all_preds.cpu().numpy()
        zs_all_labels = zs_all_labels.cpu().numpy()       

        self.cm, self.score = util.save_confusion_matrix(target=zs_all_labels, 
                                    pred=zs_all_preds, 
                                    class_names=[f"class{i}" for i in range(len(np.unique(zs_all_labels)))],
                                    save_path=os.path.join(self.logs_dir, f"zs_cm_{epoch}.png"))


    def save(self, epoch : int) -> None:
        if self.zs_ret:
            _, zs_test_acc_val, _, _, _ = self.zs_ret
            zs_save_best = zs_test_acc_val >= self.zs_test_acc_best
            if zs_save_best: 
                self.zs_test_acc_best = zs_test_acc_val   # update best zero-shot model
                self.zs_score_best = self.score

        if self.gzss_ret and self.gzsu_ret:
            _, gzss_test_acc_val, _, _, _ = self.gzss_ret
            _, gzsu_test_acc_val, _, _, _ = self.gzsu_ret
        
            gzshm_test_acc_val = util.hm(gzss_test_acc_val, gzsu_test_acc_val)
            gzs_save_best = gzshm_test_acc_val >= self.gzs_test_acc_best[2]
            if gzs_save_best: 
                self.gzs_test_acc_best[0] = gzss_test_acc_val   # update best generalized zero-shot model on seen part
                self.gzs_test_acc_best[1] = gzsu_test_acc_val   # update best generalized zero-shot model on unseen part
                self.gzs_test_acc_best[2] = gzshm_test_acc_val  # update best generalized zero-shot model based on HM

        if self.zs_ret and self.gzss_ret and self.gzsu_ret:
            # torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/model_{epoch}_{self.zs_test_acc_best}.pth")
            # torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/model_{epoch}_s={self.gzs_test_acc_best[0]}_u={self.gzs_test_acc_best[1]}_hm={self.gzs_test_acc_best[2]}.pth")
            if zs_save_best:
                # remove all previous best models
                [os.remove(f) for f in glob.glob(os.path.join(self.checkpoints_dir, "*best_zero_shot_model_*"))]
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"best_zero_shot_model_{self.zs_test_acc_best}.pth"))

            if gzs_save_best:
                # remove all previous best models
                [os.remove(f) for f in glob.glob(os.path.join(self.checkpoints_dir, "*best_generalized_zero_shot_model_*"))]
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"best_generalized_zero_shot_model_s={self.gzs_test_acc_best[0]}_u={self.gzs_test_acc_best[1]}_hm={self.gzs_test_acc_best[2]}.pth"))
