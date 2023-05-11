#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# 开始搭建网络
class NN(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)  #其实中间应该还有一层，self.fc2 = nn.linear(layer_1,layer_1)
        self.fc3 = nn.Linear(layer_1, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = self.fc3(x)
        return x
'''
这个__init__和forward是两个函数，__init__中主要是初始化一些内部需要用到的state，forward在这里没有具体实现，是需要在各个子类中实现的。
在init函数中，self.fc1 = nn.Linear(784, layer_1)
这个784指的是输入层神经元个数，因为像素大小为28*28，总共784个。
这个layer_1应该是一个被赋值的变量，是隐藏层神经元个数。
在self.fc3 = nn.Linear(layer_1, 10)
layer_1仍然指的是隐藏层神经元个数，第二个10，指的是分的类别，数字分别是从0到9
根据这个init函数，可以看出，对于self.fc1和fc3，隐藏层的神经元个数都是一样的， 所以可以认为是全连接神经网络。
'''
# class CNN_CIFAR(torch.nn.Module):
#   """Model Used by the paper introducing FedAvg"""
#   def __init__(self):
#        super(CNN_CIFAR, self).__init__()
#        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=(3,3))
#        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
#        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
#
#        self.fc1 = nn.Linear(4*4*64, 64)
#        self.fc2 = nn.Linear(64, 10)
#
#   def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x=self.conv3(x)
#        x = x.view(-1, 4*4*64)
#
#        x = F.relu(self.fc1(x))
#
#        x = self.fc2(x)
#        return x


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""  #FedAvg的CNN模型，可以直接拿来用

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


def load_model(dataset, seed): # 加载神经网络的一个函数。根据不同的数据集选择不同的神经网络

    torch.manual_seed(seed)

    if dataset == "MNIST_shard" or dataset == "MNIST_iid":
        model = NN(50, 10)  # 如果是MNIST，使用神经网络

    elif dataset[:7] == "CIFAR10":
        #        model = CNN_CIFAR()
        model = CNN_CIFAR_dropout()  # 如果是CIFAR10，使用CNN

    return model
