# coding=utf-8

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import copy
import torchvision
from torchvision import transforms as T
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
# from data import VehicleID_MC, VehicleID_All, id2name
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.font_manager import *
from collections import defaultdict

from InitRepNet import InitRepNet

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']


# --------------------------------------
# VehicleID用于MDNet
class VehicleID_All(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train_all.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test_all.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        # 打开vid2TrainID和trainID2Vid映射
        vid2TrainID_path = root + '/attribute/vid2TrainID.pkl'
        trainID2Vid_path = root + '/attribute/trainID2Vid.pkl'
        if not (os.path.isfile(vid2TrainID_path) \
                and os.path.isfile(trainID2Vid_path)):
            print('=> [Err]: invalid vid, train_id mapping file path.')

        with open(vid2TrainID_path, 'rb') as fh_1, \
                open(trainID2Vid_path, 'rb') as fh_2:
            self.vid2TrainID = pickle.load(fh_1)
            self.trainID2Vid = pickle.load(fh_2)

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)

                    tr_id = self.vid2TrainID[int(line[3])]
                    label = np.array([int(line[1]),
                                      int(line[2]),
                                      int(tr_id)], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


# Vehicle ID用于车型和颜色的多标签分类
class VehicleID_MC(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)
                    label = np.array([int(line[1]), int(line[2])], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


class FocalLoss(nn.Module):
    """
    Focal loss: focus more on hard samples
    """

    def __init__(self,
                 gamma=0,
                 eps=1e-7):
        """
        :param gamma:
        :param eps:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:
        """
        log_p = self.ce(input, target)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()


# -----------------------------------FC layers
class ArcFC(nn.Module):
    r"""
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output_layer sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        ArcMargin
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print('=> in dim: %d, out dim: %d' % (self.in_features, self.out_features))

        self.s = s
        self.m = m

        # 根据输入输出dim确定初始化权重
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # L2 normalize and calculate cosine
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # phi: cos(θ+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # ----- whether easy margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)  # device='cuda'
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output_layer)

        return output


# ---------- Mixed Difference Network Structure base on vgg16
class RepNet(torch.nn.Module):
    def __init__(self,
                 out_ids,
                 out_attribs):
        """
        Network definition
        :param out_ids:
        :param out_attribs:
        """
        super(RepNet, self).__init__()

        self.out_ids, self.out_attribs = out_ids, out_attribs
        print('=> out_ids: %d, out_attribs: %d' % (self.out_ids, self.out_attribs))

        # Conv1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (0)
        self.conv1_2 = torch.nn.ReLU(inplace=True)  # (1)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (2)
        self.conv1_4 = torch.nn.ReLU(inplace=True)  # (3)
        self.conv1_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (4)

        self.conv1 = torch.nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4,
            self.conv1_5
        )

        # Conv2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (5)
        self.conv2_2 = torch.nn.ReLU(inplace=True)  # (6)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (7)
        self.conv2_4 = torch.nn.ReLU(inplace=True)  # (8)
        self.conv2_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (9)

        self.conv2 = torch.nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
            self.conv2_4,
            self.conv2_5
        )

        # Conv3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (10)
        self.conv3_2 = torch.nn.ReLU(inplace=True)  # (11)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (12)
        self.conv3_4 = torch.nn.ReLU(inplace=True)  # (13)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (14)
        self.conv3_6 = torch.nn.ReLU(inplace=True)  # (15)
        self.conv3_7 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (16)

        self.conv3 = torch.nn.Sequential(
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4,
            self.conv3_5,
            self.conv3_6,
            self.conv3_7
        )

        # Conv4_1
        self.conv4_1_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (17)
        self.conv4_1_2 = torch.nn.ReLU(inplace=True)  # (18)
        self.conv4_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (19)
        self.conv4_1_4 = torch.nn.ReLU(inplace=True)  # (20)
        self.conv4_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (21)
        self.conv4_1_6 = torch.nn.ReLU(inplace=True)  # (22)
        self.conv4_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (23)

        self.conv4_1 = torch.nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2,
            self.conv4_1_3,
            self.conv4_1_4,
            self.conv4_1_5,
            self.conv4_1_6,
            self.conv4_1_7
        )

        # Conv4_2
        self.conv4_2_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (17)
        self.conv4_2_2 = torch.nn.ReLU(inplace=True)  # (18)
        self.conv4_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (19)
        self.conv4_2_4 = torch.nn.ReLU(inplace=True)  # (20)
        self.conv4_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (21)
        self.conv4_2_6 = torch.nn.ReLU(inplace=True)  # (22)
        self.conv4_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (23)

        self.conv4_2 = torch.nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2,
            self.conv4_2_3,
            self.conv4_2_4,
            self.conv4_2_5,
            self.conv4_2_6,
            self.conv4_2_7
        )

        # Conv5_1
        self.conv5_1_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (24)
        self.conv5_1_2 = torch.nn.ReLU(inplace=True)  # (25)
        self.conv5_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (26)
        self.conv5_1_4 = torch.nn.ReLU(inplace=True)  # (27)
        self.conv5_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (28)
        self.conv5_1_6 = torch.nn.ReLU(inplace=True)  # (29)
        self.conv5_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (30)

        self.conv5_1 = torch.nn.Sequential(
            self.conv5_1_1,
            self.conv5_1_2,
            self.conv5_1_3,
            self.conv5_1_4,
            self.conv5_1_5,
            self.conv5_1_6,
            self.conv5_1_7
        )

        # Conv5_2
        self.conv5_2_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (24)
        self.conv5_2_2 = torch.nn.ReLU(inplace=True)  # (25)
        self.conv5_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (26)
        self.conv5_2_4 = torch.nn.ReLU(inplace=True)  # (27)
        self.conv5_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (28)
        self.conv5_2_6 = torch.nn.ReLU(inplace=True)  # (29)
        self.conv5_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (30)

        self.conv5_2 = torch.nn.Sequential(
            self.conv5_2_1,
            self.conv5_2_2,
            self.conv5_2_3,
            self.conv5_2_4,
            self.conv5_2_5,
            self.conv5_2_6,
            self.conv5_2_7
        )

        # FC6_1
        self.FC6_1_1 = torch.nn.Linear(in_features=25088,
                                       out_features=4096,
                                       bias=True)  # (0)
        self.FC6_1_2 = torch.nn.ReLU(inplace=True)  # (1)
        self.FC6_1_3 = torch.nn.Dropout(p=0.5)  # (2)
        self.FC6_1_4 = torch.nn.Linear(in_features=4096,
                                       out_features=4096,
                                       bias=True)  # (3)
        self.FC6_1_5 = torch.nn.ReLU(inplace=True)  # (4)
        self.FC6_1_6 = torch.nn.Dropout(p=0.5)  # (5)

        self.FC6_1 = torch.nn.Sequential(
            self.FC6_1_1,
            self.FC6_1_2,
            self.FC6_1_3,
            self.FC6_1_4,
            self.FC6_1_5,
            self.FC6_1_6
        )

        # FC6_2
        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)

        self.FC6_2 = torch.nn.Sequential(
            self.FC6_2_1,
            self.FC6_2_2,
            self.FC6_2_3,
            self.FC6_2_4,
            self.FC6_2_5,
            self.FC6_2_6
        )

        # FC7_1
        self.FC7_1 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)  # (6): 4096, 1000

        # FC7_2
        self.FC7_2 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)  # (6): 4096, 1000

        # ------------------------------ extra layers: FC8 and FC9
        self.FC_8 = torch.nn.Linear(in_features=2000,  # 2048
                                    out_features=1024)  # 1024

        # attribute classifiers: out_attribs to be decided
        self.attrib_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=out_attribs)

        # Arc FC layer for branch_2 and branch_3
        self.arc_fc_br2 = ArcFC(in_features=1000,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)

        # construct branches
        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_1_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_1,
            self.conv5_1,
        )

        self.branch_1_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC7_1
        )

        self.branch_1 = torch.nn.Sequential(
            self.branch_1_feats,
            self.branch_1_fc
        )

        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )

        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_2,
            self.FC7_2
        )

        self.branch_2 = torch.nn.Sequential(
            self.branch_2_feats,
            self.branch_2_fc
        )

    def forward(self,
                X,
                branch,
                label=None):
        """
        :param X:
        :param branch:
        :param label:
        :return:
        """
        # batch size
        N = X.size(0)

        if branch == 1:  # train attributes classification
            X = self.branch_1_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            X = self.attrib_classifier(X)

            assert X.size() == (N, self.out_attribs)

            return X

        elif branch == 2:  # get vehicle fine-grained feature
            if label is None:
                print('=> label is None.')
                return None
            X = self.branch_2_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_2_fc(X)

            assert X.size() == (N, 1000)

            X = self.arc_fc_br2.forward(input=X, label=label)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 3:  # overall: combine branch_1 and branch_2
            if label is None:
                print('=> label is None.')
                return None
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

            # feature fusion
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)

            # connect to FC8: output 1024 dim feature vector
            X = self.FC_8(fusion_feats)

            # connect to classifier: arc_fc_br3
            X = self.arc_fc_br3.forward(input=X, label=label)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 4:  # test pre-trained weights
            # extract features
            X = self.branch_1_feats(X)

            # flatten and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            return X

        elif branch == 5:
            # 前向运算提取用于Vehicle ID的特征向量
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

            # feature fusion
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)

            # connect to FC8: output 1024 dim feature vector
            X = self.FC_8(fusion_feats)

            assert X.size() == (N, 1024)

            return X

        else:
            print('=> invalid branch')
            return None


# --------------------------------------- methods
def get_predict_mc(output):
    """
    softmax归一化,然后统计每一个标签最大值索引
    :param output:
    :return:
    """
    # 计算预测值
    output = output.cpu()  # 从GPU拷贝出来
    pred_model = output[:, :250]
    pred_color = output[:, 250:]

    model_idx = pred_model.max(1, keepdim=True)[1]
    color_idx = pred_color.max(1, keepdim=True)[1]

    # 连接pred
    pred = torch.cat((model_idx, color_idx), dim=1)
    return pred


def count_correct(pred, label):
    """
    :param output:
    :param label:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if torch.equal(one, two):
            correct_num += 1
    return correct_num


def count_attrib_correct(pred, label, idx):
    """
    :param pred:
    :param label:
    :param idx:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if one[idx] == two[idx]:
            correct_num += 1
    return correct_num


# @TODO: 可视化分类结果...

def ivt_tensor_img(input,
                   title=None):
    """
    Imshow for Tensor.
    """
    input = input.numpy().transpose((1, 2, 0))

    # 转变数组格式 RGB图像格式：rows * cols * channels
    # 灰度图则不需要转换，只有(rows, cols)而不是（rows, cols, 1）
    # (3, 228, 906)   #  (228, 906, 3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 去标准化，对应transforms
    input = std * input + mean

    # 修正 clip 限制inp的值，小于0则=0，大于1则=1
    output = np.clip(input, 0, 1)

    # plt.imshow(input)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

    return output


def viz_results(resume,
                data_root):
    """
    :param resume:
    :param data_root:
    :return:
    """
    color_dict = {'black': u'黑色',
                  'blue': u'蓝色',
                  'gray': u'灰色',
                  'red': u'红色',
                  'sliver': u'银色',
                  'white': u'白色',
                  'yellow': u'黄色'}

    test_set = VehicleID_All(root=data_root,
                             transforms=None,
                             mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

    net = RepNet(out_ids=10086,
                 out_attribs=257).to(device)
    print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 测试模式
    net.eval()

    # 加载类别id映射和类别名称
    modelID2name_path = data_root + '/attribute/modelID2name.pkl'
    colorID2name_path = data_root + '/attribute/colorID2name.pkl'
    trainID2Vid_path = data_root + '/attribute/trainID2Vid.pkl'
    if not (os.path.isfile(modelID2name_path) and \
            os.path.isfile(colorID2name_path) and \
            os.path.isfile((trainID2Vid_path))):
        print('=> [Err]: invalid file.')
        return

    with open(modelID2name_path, 'rb') as fh_1, \
            open(colorID2name_path, 'rb') as fh_2, \
            open(trainID2Vid_path, 'rb') as fh_3:
        modelID2name = pickle.load(fh_1)
        colorID2name = pickle.load(fh_2)
        trainID2Vid = pickle.load(fh_3)

    # 测试
    print('=> testing...')
    for i, (data, label) in enumerate(test_loader):
        # 放入GPU.
        data, label = data.to(device), label.to(device).long()

        # 前向运算: 预测车型、车身颜色
        output_attrib = net.forward(X=data,
                                    branch=1,
                                    label=None)
        pred_mc = get_predict_mc(output_attrib).cpu()[0]
        pred_m_id, pred_c_id = pred_mc[0].item(), pred_mc[1].item()
        pred_m_name = modelID2name[pred_m_id]
        pred_c_name = colorID2name[pred_c_id]

        # 前向运算: 预测Vehicle ID
        output_id = net.forward(X=data,
                                branch=3,
                                label=label[:, 2])
        _, pred_tid = torch.max(output_id, 1)
        pred_tid = pred_tid.cpu()[0].item()
        pred_vid = trainID2Vid[pred_tid]

        # 获取实际result
        img_path = test_loader.dataset.imgs_path[i]
        img_name = os.path.split(img_path)[-1][:-4]

        result = label.cpu()[0]
        res_m_id, res_c_id, res_vid = result[0].item(), result[1].item(), \
                                      trainID2Vid[result[2].item()]
        res_m_name = modelID2name[res_m_id]
        res_c_name = colorID2name[res_c_id]

        # 图像标题
        title = 'pred: ' + pred_m_name + ' ' + color_dict[pred_c_name] \
                + ', vehicle ID ' + str(pred_vid) \
                + '\n' + 'resu: ' + res_m_name + ' ' + color_dict[res_c_name] \
                + ', vehicle ID ' + str(res_vid)
        print('=> result: ', title)

        # 绘图
        img = ivt_tensor_img(data.cpu()[0])
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.show()


def gen_test_pairs(test_txt,
                   dst_dir,
                   num=10000):
    """
    生成测试pair数据: 一半positive，一半negative
    :param test_txt:
    :return:
    """
    if not os.path.isfile(test_txt):
        print('[Err]: invalid file.')
        return
    print('=> genarating %d samples...' % num)

    with open(test_txt, 'r') as f_h:
        valid_list = f_h.readlines()
        print('=> %s loaded.' % test_txt)

        # 映射: img_name => cls_id
        valid_dict = {x.strip().split()[0]: int(x.strip().split()[3]) for x in valid_list}

        # 映射: cls_id => img_list
        inv_dict = defaultdict(list)
        for k, v in valid_dict.items():
            inv_dict[v].append(k)

        # 统计样本数不少于2的id
        big_ids = [k for k, v in inv_dict.items() if len(v) > 1]

    # 添加测试样本
    pair_set = set()
    while len(pair_set) < num:
        if random.random() <= 0.7:  # positive
            # 随机从big_ids中选择一个
            pick_id = random.sample(big_ids, 1)[0]  # 不放回抽取

            anchor = random.sample(inv_dict[pick_id], 1)[0]
            positive = random.choice(inv_dict[pick_id])
            while positive == anchor:
                positive = random.choice(inv_dict[pick_id])

            pair_set.add(anchor + '\t' + positive + '\t1')
        else:  # negative
            pick_id_1 = random.sample(big_ids, 1)[0]  # 不放回抽取
            pick_id_2 = random.sample(big_ids, 1)[0]  # 不放回抽取
            while pick_id_2 == pick_id_1:
                pick_id_2 = random.sample(big_ids, 1)[0]
            assert pick_id_2 != pick_id_1
            anchor = random.choice(inv_dict[pick_id_1])
            negative = random.choice(inv_dict[pick_id_2])

            pair_set.add(anchor + '\t' + negative + '\t0')
    print(list(pair_set)[:5])
    print(len(pair_set))

    # 序列化pair_set到dst_dir
    pair_set_f_path = dst_dir + '/' + 'pair_set_vehicle.txt'
    with open(pair_set_f_path, 'w') as f_h:
        for x in pair_set:
            f_h.write(x + '\n')
    print('=> %s generated.' % pair_set_f_path)


# 获取每张测试图片对应的特征向量
def gen_feature_map(resume,
                    imgs_path,
                    batch_size=16):
    """
    根据图相对生成每张图象的特征向量, 映射: img_name => img_feature vector
    :param resume:
    :param imgs_path:
    :return:
    """
    net = RepNet(out_ids=10086,
                 out_attribs=257).to(device)
    print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 图像数据变换
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    # load model, image and forward
    data, features = None, None
    for i, img_path in tqdm(enumerate(imgs_path)):
        # load image
        img = Image.open(img_path)

        # tuen to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # image data transformations
        img = transforms(img)
        img = img.view(1, 3, 224, 224)

        if data is None:
            data = img
        else:
            data = torch.cat((data, img), dim=0)

        if data.shape[0] % batch_size == 0 or i == len(imgs_path) - 1:

            # collect a batch of image data
            data = data.to(device)

            output = net.forward(X=data,
                                 branch=5,
                                 label=None)

            batch_features = output.data.cpu().numpy()
            if features is None:
                features = batch_features
            else:
                features = np.vstack((features, batch_features))

            # clear a batch of images
            data = None

    # generate feature map
    feature_map = {}
    for i, img_path in enumerate(imgs_path):
        feature_map[img_path] = features[i]

    print('=> feature map size: %d' % (len(feature_map)))
    return feature_map


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    """
    :param y_score:
    :param y_true:
    :return:
    """
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        print('=> th: %.3f, acc: %.3f' % (th, acc))

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


# 统计阈值和准确率: Vehicle ID数据集
def get_th_acc_VID(resume,
                   pair_set_txt,
                   img_dir,
                   batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param img_dir:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            pair = line.strip().split()

            imgs_path.append(img_dir + '/' + pair[0] + '.jpg')
            imgs_path.append(img_dir + '/' + pair[1] + '.jpg')

            pairs.append(pair)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # generate feature dict
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_dir + '/' + pair[0] + '.jpg'
        img_path_2 = img_dir + '/' + pair[1] + '.jpg'
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


# 统计阈值和准确率: Car Match数据集
def test_car_match_data(resume,
                        pair_set_txt,
                        img_root,
                        batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = line.strip().split()

            imgs_path.append(img_root + '/' + line[0])
            imgs_path.append(img_root + '/' + line[1])

            pairs.append(line)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # 计算特征向量字典
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    # 计算所有pair的sim
    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_root + '/' + pair[0]
        img_path_2 = img_root + '/' + pair[1]
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


def test_accuracy(net, data_loader):
    """
    测试VehicleID分类在测试集上的准确率
    :param net:
    :param data_loader:
    :return:
    """
    net.eval()  # 测试模式,前向计算

    num_correct = 0
    num_total = 0

    # 每个属性的准确率
    num_model = 0
    num_color = 0
    total_time = 0.0

    print('=> testing...')
    for data, label in data_loader:
        # 放入GPU.
        data, label = data.to(device), label.to(device).long()

        # 前向运算, 预测Vehicle ID
        output = net.forward(X=data,
                             branch=3,
                             label=label[:, 2])

        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        _, pred = torch.max(output.data, 1)
        batch_correct = (pred == label[:, 2]).sum().item()
        num_correct += batch_correct

    # test-set总的统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    print('=> test accuracy: {:.3f}%'.format(accuracy))

    return accuracy


def test_mc_accuracy(net,
                     data_loader):
    """
    :param net:
    :param data_loader:
    :return:
    """
    net.eval()  # 测试模式,前向计算

    num_correct = 0
    num_total = 0

    # 每个属性的准确率
    num_model = 0
    num_color = 0
    total_time = 0.0

    print('=> testing...')
    for data, label in data_loader:
        # 放入GPU.
        data, label = data.to(device), label.to(device)

        # 将label转化为cpu, long
        label = label.cpu().long()

        # 前向运算, 预测
        output = net.forward(X=data, branch=1)  # 默认在device(GPU)中推理运算
        pred = get_predict_mc(output)  # 返回的pred存在于host端

        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        num_correct += count_correct(pred, label)

        # 统计各属性正确率
        num_model += count_attrib_correct(pred, label, 0)
        num_color += count_attrib_correct(pred, label, 1)

    # 总统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    model_acc = 100.0 * float(num_model) / float(num_total)
    color_acc = 100.0 * float(num_color) / float(num_total)

    print('=> test accuracy: {:.3f}%, RAModel accuracy: {:.3f}%, '
          'color accuracy: {:.3f}%'.format(
        accuracy, model_acc, color_acc))
    return accuracy


def train_mc(freeze_feature,
             resume=None):
    """
    训练RepNet: RAModel and color multi-label classification
    :param freeze_feature:
    :return:
    """
    net = RepNet(out_ids=10086,
                 out_attribs=257).to(device)
    print('=> Mix difference network:\n', net)

    # 是否从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))  # 加载模型
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_MC(root='/mnt/diskb/even/VehicleID_V1.0',
                             transforms=None,
                             mode='train')
    test_set = VehicleID_MC(root='/mnt/diskb/even/VehicleID_V1.0',
                            transforms=None,
                            mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=2)

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    # 优化函数
    if freeze_feature:  # 锁住特征提取层，仅打开FC层
        optimizer = torch.optim.SGD(net.branch_1_fc.parameters(),
                                    lr=1e-3,
                                    momentum=9e-1,
                                    weight_decay=1e-8)
        for param in net.branch_1_feats.parameters():
            param.requires_grad = False
        print('=> optimize only FC layers.')
    else:  # 打开所有参数
        optimizer = torch.optim.SGD(net.branch_1.parameters(),
                                    lr=1e-3,
                                    momentum=9e-1,
                                    weight_decay=1e-8)
        print('=> optimize all layers.')

    # 开始训练
    print('\nTraining...')
    net.train()  # train模式

    best_acc = 0.0
    best_epoch = 0

    print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
    for epoch in range(50):
        epoch_loss = []
        num_correct = 0
        num_total = 0

        for data, label in train_loader:  # 遍历每一个batch
            # ------------- 放入GPU
            data, label = data.to(device), label.to(device).long()

            # ------------- 清空梯度
            optimizer.zero_grad()

            # ------------- 前向计算
            output = net.forward(X=data, branch=1)

            # 计算loss
            loss_m = loss_func(output[:, :250], label[:, 0])
            loss_c = loss_func(output[:, 250:], label[:, 1])
            loss = loss_m + loss_c

            # ------------- 统计
            epoch_loss.append(loss.item())

            # 统计样本数量
            num_total += label.size(0)

            # 统计训练数据正确率
            pred = get_predict_mc(output)
            label = label.cpu().long()
            num_correct += count_correct(pred=pred, label=label)

            # ------------- 反向运算
            loss.backward()
            optimizer.step()

        # 计算训练集准确度
        train_acc = 100.0 * float(num_correct) / float(num_total)

        # 计算测试集准确度
        test_acc = test_mc_accuracy(net=net,
                                    data_loader=test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            # 保存模型权重
            model_save_name = 'epoch_' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(),
                       '/mnt/diskb/even/MDNet_ckpt_br1/' + model_save_name)
            print('<= {} saved.'.format(model_save_name))

        print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
              (epoch + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))


def train(resume):
    """
    :param resume:
    :return:
    """
    # net = RepNet(out_ids=10086,
    #              out_attribs=257).to(device)

    vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
    net = InitRepNet(vgg_orig=vgg16_pretrain,
                     out_ids=10086,
                     out_attribs=257).to(device)

    print('=> Mix difference network:\n', net)

    # whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))  # 加载模型
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_All(root='/mnt/diskb/even/VehicleID_V1.0',
                              transforms=None,
                              mode='train')
    test_set = VehicleID_All(root='/mnt/diskb/even/VehicleID_V1.0',
                             transforms=None,
                             mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=4)

    # loss function
    loss_func_1 = torch.nn.CrossEntropyLoss().to(device)
    loss_func_2 = FocalLoss(gamma=2).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=1e-3,
                                momentum=9e-1,
                                weight_decay=1e-8)
    print('=> optimize all layers.')

    # start to train
    print('\nTraining...')
    net.train()  # train模式

    best_acc = 0.0
    best_epoch = 0

    print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
    for epoch_i in range(30):

        epoch_loss = []
        num_correct = 0
        num_total = 0
        for batch_i, (data, label) in enumerate(train_loader):  # 遍历每一个batch
            # ------------- put data to device
            data, label = data.to(device), label.to(device).long()

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass of 3 branches
            output_1 = net.forward(X=data, branch=1, label=None)
            output_2 = net.forward(X=data, branch=2, label=label[:, 2])
            output_3 = net.forward(X=data, branch=3, label=label[:, 2])

            # ------------- calculate loss
            # branch1 loss
            loss_m = loss_func_1(output_1[:, :250], label[:, 0])  # vehicle model
            loss_c = loss_func_1(output_1[:, 250:], label[:, 1])  # vehicle color
            loss_br1 = loss_m + loss_c

            # branch2 loss
            loss_br2 = loss_func_2(output_2, label[:, 2])

            # branch3 loss: Vehicle ID classification
            loss_br3 = loss_func_2(output_3, label[:, 2])

            # 加权计算总loss
            loss = 0.5 * loss_br1 + 0.5 * loss_br2 + 1.0 * loss_br3

            # ------------- statistics
            epoch_loss.append(loss.cpu().item())

            # count samples
            num_total += label.size(0)

            # statistics of correct number
            _, pred = torch.max(output_3.data, 1)
            batch_correct = (pred == label[:, 2]).sum().item()
            batch_acc = float(batch_correct) / float(label.size(0))
            num_correct += batch_correct

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            iter_count = epoch_i * len(train_loader) + batch_i

            # output batch accuracy
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>4d}/{:>4d}'
                      ', total_iter {:>6d} '
                      '| loss {:>5.3f} | accuracy {:>.3%}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item(),
                              batch_acc))

        # total epoch accuracy
        train_acc = float(num_correct) / float(num_total)
        print('=> epoch {} | average loss: {:.3f} | average accuracy: {:>.3%}'
              .format(epoch_i + 1,
                      float(sum(epoch_loss)) / float(len(epoch_loss)),
                      train_acc))

        # calculate test-set accuracy
        test_acc = test_accuracy(net=net,
                                 data_loader=test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch_i + 1

            # save model weights
            model_save_name = 'epoch_' + str(epoch_i + 1) + '.pth'
            torch.save(net.state_dict(),
                       '/mnt/diskb/even/MDNet_ckpt_all/' + model_save_name)
            print('<= {} saved.'.format(model_save_name))

        print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
              (epoch_i + 1,
               sum(epoch_loss) / len(epoch_loss),
               train_acc * 100.0,
               test_acc))
    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))


if __name__ == '__main__':
    # test_init_weight()

    # train_mc(freeze_feature=False,
    #          resume='/mnt/diskb/even/MDNet_ckpt_br1/epoch_16.pth')

    # train(resume='/mnt/diskb/even/MDNet_ckpt_br1/epoch_16.pth')
    train(resume=None)  # 从头开始训练

    # train(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_24_bk.pth')

    # -----------------------------------
    # viz_results(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_12.pth',
    #             data_root='/mnt/diskb/even/VehicleID_V1.0')

    # test_car_match_data(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_10.pth',
    #                     pair_set_txt='/mnt/diskc/even/Car_DR/ArcFace_pytorch/data/pair_set_car.txt',
    #                     img_root='/mnt/diskc/even/CarReIDCrop',  # CarReID_data
    #                     batch_size=16)

    # get_th_acc_VID(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_10.pth',
    #                pair_set_txt='/mnt/diskb/even/VehicleID_V1.0/attribute/pair_set_vehicle.txt',
    #                img_dir='/mnt/diskb/even/VehicleID_V1.0/image',
    #                batch_size=16)

    print('=> Done.')
