import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from dataloaders.dataloader import initDataloader
from modeling.net import DRA
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from modeling.layers import build_criterion
import random

import matplotlib.pyplot as plt
from datetime import datetime
import copy
from torch.utils.tensorboard import SummaryWriter

WEIGHT_DIR = './weights'
writer = SummaryWriter('runs/logs')

class Trainer(object):

    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader, _= initDataloader.build(args, **kwargs)
        if self.args.total_heads == 4:
            temp_args = copy.deepcopy(args)
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _, _ = initDataloader.build(temp_args, **kwargs)
            self.ref = iter(self.ref_loader)

            # for save features
            temp_args1 = copy.deepcopy(args)
            temp_args1.sftransform = True
            _, _, self.sf_loader = initDataloader.build(temp_args1, **kwargs)

        self.model = DRA(args, backbone=self.args.backbone)

        if self.args.pretrain_dir != None:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target==0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
        return targets

    def training(self, epoch):
        train_loss = 0.0
        class_loss = list()
        for j in range(self.args.total_heads):
            class_loss.append(0.0)
        self.model.train()
        self.scheduler.step()
        tbar = tqdm(self.train_loader)
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            outputs, tmpLoss = self.model(image, target)     # 四个头，每个头为每张图输出一个分数
            targets = self.generate_target(target)

            losses = list()
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs[i], dim=1)
                    losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                else:
                    losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))
            losses.append(tmpLoss.view(-1, 1))
            loss = torch.cat(losses)
            loss = torch.sum(loss)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            # tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (idx + 1)))
            tbar.set_description(f'Epoch:{epoch}, Train loss: {train_loss / (idx + 1):.3f}, h: {losses[0].item():.3f}, s: {losses[1].item():.3f}, p: {losses[2].item():.3f}, c: {losses[3].item():.3f}, proto: {losses[4].item():.3f}')
            writer.add_scalar('train_loss', loss.item(), epoch * len(self.train_loader) + sample['image'].shape[0])

    def saveFeatures(self):
        self.model.eval()
        # train_features
        tbar = tqdm(self.sf_loader)
        train_features = []
        labels = []
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target == 2] = 0
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            embeds = self.model(image, target, self.args.epochs)     # 四个头，每个头为每张图输出一个分数
            train_features.append(embeds.detach().cpu().numpy())
            # labels.append(target.detach().cpu().numpy())
            
            tbar.set_description(f'Epoch:{self.args.epochs}, Saving train_features...')
        # 将特征和标签转换为numpy数组
        train_features = np.concatenate(train_features, axis=0)
        # labels = np.concatenate(labels, axis=0)
        # 保存特征和标签
        np.save(f'{self.args.experiment_dir}/train_features.npy', train_features)
        # np.save(f'{self.args.experiment_dir}/labels.npy', labels)
        
        # test_features
        tbar = tqdm(self.test_loader)
        test_features = []
        test_labels = []
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target == 2] = 0
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            embeds = self.model(image, target, self.args.epochs)     # 四个头，每个头为每张图输出一个分数
            test_features.append(embeds.detach().cpu().numpy())
            test_labels.append(target.detach().cpu().numpy())
            
            tbar.set_description(f'Epoch:{self.args.epochs}, Saving test_features...')
        # 将特征和标签转换为numpy数组
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        # 保存特征和标签
        np.save(f'{self.args.experiment_dir}/test_features.npy', train_features)
        np.save(f'{self.args.experiment_dir}/test_labels.npy', labels)

    def normalization(self, data):
        # f_norm = np.linalg.norm(data)
        # # 进行F-norm归一化
        # normalized_data = data / f_norm
        # return normalized_data
        return data
    
    def minMaxnorm(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data
        

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        class_pred = list()
        assistPred = np.array([])
        for i in range(self.args.total_heads):
            class_pred.append(np.array([]))
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            with torch.no_grad():
                outputs, assistOut = self.model(image, target)
                assistPred = np.append(assistPred, assistOut.data.cpu().numpy())
                targets = self.generate_target(target, eval=True)

                losses = list()
                for i in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = losses[0]
                for i in range(1, self.args.total_heads):
                    loss += losses[i]

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            for i in range(self.args.total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)
            total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])
        total_pred = total_pred + assistPred
        # total_pred = self.minMaxnorm(total_pred)

        total_roc, total_pr = aucPerformance(total_pred, total_target)
        # total_roc_A, total_pr_A = aucPerformance(assistPred, total_target)

        
        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score, assistScore in zip(total_target, total_pred, assistPred):
                # w.write(f'{str(label)}\t\t{score:.3f}\t\t{assistScore:.3f}\n')
                w.write(f'{str(label)}\t\t{score:.3f}\n')
            w.write("AUC-ROC: " + str(total_roc) + "\nAUC-PR: " + str(total_pr))
            # w.write("AUC-ROC_A: " + str(total_roc_A) + "\nAUC-PR_A: " + str(total_pr_A))

        normal_mask = total_target == 0
        outlier_mask = total_target == 1
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red')
        plt.ylabel("Anomaly score")
        plt.savefig(args.experiment_dir + "/vis.png")

        printROCcurve(total_pred, total_target)
        return total_roc, total_pr

    def save_weights(self, filename):
        # if not os.path.exists(WEIGHT_DIR):
        #     os.makedirs(WEIGHT_DIR)
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

    def load_weights(self, filename):
        path = os.path.join(WEIGHT_DIR, filename)
        self.model.load_state_dict(torch.load(path))

    def init_network_weights_from_pretraining(self):

        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

def printROCcurve(total_pred, total_target):
    fpr, tpr, thresholds = roc_curve(total_target, total_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2  # 线宽
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(args.experiment_dir + "/roc_curve.png")

def getExperPath(args):
    subName = ""
    if args.dataset == "mvtecad":
        if args.know_class == "None":
            subName = f'general/{args.nAnomaly}/{args.dataset}/{args.classname}'
        else:
            subName = f'hard/{args.nAnomaly}/{args.dataset}/{args.classname}/{args.know_class}'
    else:
        if args.know_class == "None":
            subName = f'general/{args.nAnomaly}/{args.dataset}'
        else:
            subName = f'hard/{args.nAnomaly}/{args.dataset}/{args.know_class}'

    path = f'{args.experiment_dir}/{subName}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default=None, help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="")
    parser.add_argument('--classname', type=str, default='carpet', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="dataset root")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")

    parser.add_argument('--numProtos', type=int, default=3, help="number of protos")
    parser.add_argument('--beta', type=float, default=0.01, help="factor to update protos")
    parser.add_argument('--cdfl', type=bool, default=False, help="ablation of cdfl module")
    parser.add_argument('--sf', type=bool, default=False, help="save features")
    parser.add_argument('--sftransform', type=bool, default=False, help="when save features, dataloader's transform")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)


    argsDict = args.__dict__
    args.experiment_dir = getExperPath(args)
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Total Epoches:', trainer.args.epochs)
    trainer.model = trainer.model.to('cuda')
    trainer.criterion = trainer.criterion.to('cuda')
    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
    
    # save features
    if args.sf == True:
        trainer.saveFeatures()

    writer.close()
    trainer.eval()
    trainer.save_weights(args.savename)

