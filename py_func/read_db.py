#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

"""
通过一系列的切分规则将数据水平分布到不同的DB或table中，在通过相应的DB路由 或者 table路由规则找到需要查询的具体的DB或者table，
以进行Query操作。这里所说的“sharding”通常是指“水平切分”
shard其实就是数据碎片，就是一桶水用N个杯子装，分片之间都是独立的。

在数据库中，cluster叫做集群，node叫做节点，shard叫做分片，indices叫做索引，replicas叫做备份
"""
# 这个文件应该就是如何读取数据集，应该可以通用
"""
-------------
MNIST non-iid
-------------
"""

# 这个函数的作用应该是创建一个数据分片，主要目的是给每一个客户端都能创建一个本地数据集（这个只是一个创建数据分片实现途径）
def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`
    row_0应该是ds即数据集dataset的指标这一行
    虽然是创建一个数据分片，但是每个shard都需要有一层指标，放在row_0
    """

    row = row_0 # 定义变量名row的值为row_0，即datasets的第一行，是指标名

    shard = list() # 定义一个list，记作shard

    while len(shard) < samples: # 当数据分片shard这个list的长度小于输入进来的samples变量的值时
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard


def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test): # 给每一个客户端创建一个基于MINIST数据集的本地数据分片

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True) # 分配MINIST数据集的训练集
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True) # 分配MINIST数据集的测试集

    shards_train, shards_test = [], [] # 创建shard数据分片的训练集和测试集的空白list
    labels = [] # 创建labels的空白list

    for i in range(10): # 这个10可能与MINIST数据集的种类相关
        row_train, row_test = 0, 0 # 这个又创建了个变量row_train和row_test
        for j in range(10):
            row_train, shard_train = get_1shard(
                MNIST_train, row_train, i, samples_train
            ) # 通过调用get_1shard函数来重复生成训练数据集？分别作为row_train和shard_train
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            ) # 通过调用get_1shard函数来重复生成测试数据集，分别作为row_test和shard_test

            shards_train.append([shard_train]) # 通过调用.append函数来扩充shards_train ps：list只能通过append和insert来插入元素
            shards_test.append([shard_test]) # 通过调用.append函数来扩充shards_test

            labels += [[i]]

    X_train = np.array(shards_train) # 将创建完成的shards_train数据集变成np.array来作为X.train
    X_test = np.array(shards_test) # 将创建完成的shards_test数据集变成np.array来作为X.test

    y_train = labels # 将创建完成的labels数据集作为y_train
    y_test = y_train # y_train赋值给y_test

    folder = "./data/" # 写明数据集所在的文件夹
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl" # 这个是训练数据集的路径，因为此时我们将数据集已经按照客户端训练样本进行创建好了，然后保存成.pkl文件
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output) # pickle.dump函数作用是序列化对象，并将结果数据流写入到文件对象中

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl" # 表明测试数据集的路径
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)

# 定义个函数，create_MNIST_small_niid，用于创建基于MINIST数据集的非独立同分布数据
def create_MNIST_small_niid(
    n_clients: int,
    samples_train: list,
    samples_test: list,
    clients_digits: list,
): # 这个产生non-iid数据集的函数，主要输入为客户端的个数、用于训练的样本、用于测试的样本、客户端所拥有的的数字

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    X_train, X_test = [], []
    y_train, y_test = [], []

    for digits, n_train, n_test in zip(
        clients_digits, samples_train, samples_test
    ): # zip函数将clients_digits,samples_train,samples_test打包为一个列表

        client_samples_train, client_samples_test = [], [] # 在for循环内部，设置client_samples的训练集和测试集的空表
        client_labels_train, client_labels_test = [], [] # 设置client_labels的训练集和测试集空表

        n_train_per_shard = int(n_train / len(digits)) # 这个是训练集中每一个数据分片中的数量
        n_test_per_shard = int(n_test / len(digits)) # 这是每一个数据分片中测试集的数量

        for digit in digits:

            row_train, row_test = 0, 0
            _, shard_train = get_1shard(
                MNIST_train, row_train, digit, n_train_per_shard
            ) # get_1shard返回两个数据，一个row，一个shard，我们需要的是后者。前面不需要的就用_来容纳
            _, shard_test = get_1shard(
                MNIST_test, row_test, digit, n_test_per_shard
            )

            client_samples_train += shard_train # 客户端的训练样本通过添加训练分片来扩充
            client_samples_test += shard_test # 客户端的测试样本通过添加测试分片来扩充

            client_labels_train += [digit] * n_train_per_shard # []将digit变成一个列表，后面*n_train_per_shard表示重复这么多次，前面的client_labels_train是一个列表。列表只能加一个列表
            client_labels_test += [digit] * n_test_per_shard

        X_train.append(client_samples_train) # 将client_samples_train添加到X.train中
        X_test.append(client_samples_test)

        y_train.append(client_labels_train) # 将client_labels_train添加到y_train中
        y_test.append(client_labels_test)
    # 保存文件的形式
    folder = "./data/"
    train_path = f"MNIST_small_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((np.array(X_train), np.array(y_train)), output)

    test_path = f"MNIST_small_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((np.array(X_test), np.array(y_test)), output)

# 如果一个类表现得像一个list，要获取有多少个元素，就得用 len() 函数。要让 len() 函数工作正常，类必须提供一个特殊方法__len__()，它返回元素的个数。
class MnistShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset""" # 将MINIST的.pkl文件转化成pytorch的数据库

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file) # 调用pickle_load函数来转化dataset
            self.features = np.vstack(dataset[0][k]) # 将第1行第k+1列给输出来作为一个numpy数组，按照垂直的顺序把数组堆叠起来，作为特征层

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features) # 使用len()函数返回datasets实例的“长度”

    def __getitem__(self, idx):

        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255 # 这个应该是针对MINIST数据集，这些都是固定的程序，可以直接拿来用
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y


def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset""" # 所有参与方他们各自的数据集
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


"""
-------
CIFAR 10
Dirichilet distribution 使用狄利克雷函数来设置non-IID数据集
----
"""


def partition_CIFAR_dataset( # 定义一个CIFAR10数据集的一个分区函数
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
): # 输入分别为 数据集、文件名、是否是平衡数据、矩阵、参与方的个数、种类数、是否是训练集
    """Partition dataset into `n_clients`. 将数据集分成n_clients份，每个参与方i都有CIFAR10第k类的这块数据矩阵
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    if balanced:
        n_samples = [500] * n_clients # 如果balanced=true时，nsamples = [500,500,500,...,500] 总共n_clients个500
    elif not balanced and train:
        n_samples = (
            [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        )
    elif not balanced and not train:
        n_samples = [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10

    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)

# 这个就是基于狄利克雷分布来产生non-iid数据集了，狄利克雷分布需要一个alpha参数来调整
def create_CIFAR10_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    # 如果要调用CIFAR10数据集，在函数头需要添加import torchvision import torchvision.transforms as transforms
    CIFAR10_train = datasets.CIFAR10(  # 这个是加载用于训练的数据集
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(), # transform表示是否需要对数据进行预处理，如果是none则是不进行预处理。此处是预处理转为tensor进行计算
    )

    CIFAR10_test = datasets.CIFAR10( # 加载用于测试的数据集
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl" # 用于训练的数据集的文件名
    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    ) # 调用将CIFAR10训练数据集进行分区的函数

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl" # 用于测试的数据集的文件名
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    ) # 调用将CIFAR10测试数据集进行分区的函数

# 将前面创建好的基于CIFAR10的pkl文件转化为pytorch可用的dataset
class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255 # 这个是为了让所有的像素的值都在0，1之间，所以要除以255.
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


def clients_set_CIFAR(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


"""
---------
Upload any dataset
Puts all the function above together
---------
"""


def get_dataloaders(dataset, batch_size: int, shuffle=True):

    folder = "./data/"

    if dataset == "MNIST_iid": # 如果数据是MNIST iid数据集

        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        ) # 这个torch.utils.data.random_split是封装的划分数据集的函数，随机将一个数据集分割成给定长度的不重叠的新数据集。前面是要划分的数据集，后者是要划分的长度
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ] # torch.utils.data.DataLoader这个是数据读取的一个重要接口。这个应该是读出来训练数据后作为list_dls_train

        mnist_testset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ] # 这个读出来数据作为list_dls_test，应该是后面主要是用这个作为测试数据？

    elif dataset == "MNIST_shard": # 如果数据集是MINIST_shard
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        ) # 这个是直接调用了函数clients_set_MNIST_shard，结果作为list_dls_train

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    elif dataset == "CIFAR10_iid":
        n_clients = 100
        samples_train, samples_test = 500, 100

        CIFAR10_train = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset[:5] == "CIFAR":

        n_classes = 10
        n_clients = 100
        balanced = dataset[8:12] == "bbal"
        alpha = float(dataset[13:])

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_CIFAR10_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "wb") as output:
        pickle.dump(list_len, output)
    # 这个list_len应该是每个参与方的样本量的一个列表表示，就是这个列表中每个数值应该是每个参与方所拥有样本量的个数
    return list_dls_train, list_dls_test
