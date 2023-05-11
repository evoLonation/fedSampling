#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import product  # product用于求多个可迭代对象的笛卡尔积

from scipy.cluster.hierarchy import fcluster  # 层次聚类法
from copy import deepcopy


def get_clusters_with_alg1(n_sampled: int, weights: np.array):
    "Algorithm 1"

    epsilon = int(10 ** 10)
    # associate each client to a cluster
    augmented_weights = np.array([w * n_sampled * epsilon for w in weights])
    ordered_client_idx = np.flip(np.argsort(augmented_weights))

    n_clients = len(weights)
    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)  # 分布_聚类

    k = 0
    for client_idx in ordered_client_idx:  # idx是client的编号
        # 下面这一块有点看不懂
        while augmented_weights[client_idx] > 0:  # 如果client_idx的权重大于0，就是被抽中的client

            sum_proba_in_k = np.sum(distri_clusters[k])

            u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])

            distri_clusters[k, client_idx] = u_i
            augmented_weights[client_idx] += -u_i

            sum_proba_in_k = np.sum(distri_clusters[k])
            if sum_proba_in_k == 1 * epsilon:
                k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters


def get_similarity(grad_1, grad_2, distance_type="L1"):  # 计算距离的函数，这块看懂了！！

    if distance_type == "L1":

        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):  # zip函数的作用就是，在for循环里面，可以对应遍历grad_1 grad_2，zip()中的两个参数都必须是相同的序列对象
            norm += np.sum(np.abs(g_1 - g_2))  # 1范数，就是绝对值。
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)  # 2范数距离的python实现
        return norm

    elif distance_type == "cosine":  # cosine距离的python实现
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)


def get_gradients(sampling, global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""  # 通过对比本地工作与全局模型之间的差异来得出代表性梯度

    local_model_params = []  # 先定义一个变量是list列表数据类型，可以往这个列表里面填东西
    for model in local_models:
        local_model_params += [
            [tens.detach().numpy() for tens in list(model.parameters())]  # 这个意思是，将本地模型的梯度信息变成可以矩阵计算的数值型矩阵
        ]

    global_model_params = [
        tens.detach().numpy() for tens in list(global_m.parameters())
        # 因为global_m是作为输入的，这个时候我们要把全局模型的参数信息也转化成可以矩阵计算的数值型矩阵
    ]

    local_model_grads = []  # 先定义一个list，用于盛放本地模型的梯度信息
    for local_params in local_model_params:  # 在local_model_params这个范围里面做一个for循环
        local_model_grads += [  # 在本地模型梯度的这个list里面，不断地加
            [
                local_weights - global_weights  # 加的数值是 本地模型的权重减去全局模型的权重
                for local_weights, global_weights in zip(  # 其中，本地模型的权重和全局模型的权重是在local_params和global_model_params里面的
                local_params, global_model_params
            )
            ]
        ]

    return local_model_grads  # 通过上面的循环，可以得出local_model_grads的list。


def get_matrix_similarity_from_grads(local_model_grads, distance_type):  # 得到梯度的矩阵相似度
    """return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`"""  # 这个就是，通过多种类型的距离定义来算出两个客户端的相似矩阵

    n_clients = len(local_model_grads)  # local_model_grads是一个list，他的长度应该代表是最终抽样出来的客户端的个数

    metric_matrix = np.zeros((n_clients, n_clients))  # 先定一个方阵，维度是n_clients
    for i, j in product(range(n_clients), range(n_clients)):  # 这个product就是求多个对象的笛卡尔积。对应位置的数值进行计算

        metric_matrix[i, j] = get_similarity(  # 直接调用上面计算相似度的函数
            local_model_grads[i], local_model_grads[j], distance_type  # 在get_similarity函数里面，输入这些数据。
            # local_model_grads是一个list，i和j分别进行循环，就能把所有项之间的关系给遍历出来，然后计算相似度
        )

    return metric_matrix  # 所有循环结束后，输出的就是抽样出来的样本的


def get_matrix_similarity(global_m, local_models, distance_type):  # 这个又是一个计算相似矩阵的函数，但是是计算全局模型和本地模型的相似？

    n_clients = len(local_models)  # 其实此时还是抽样出来客户端的个数

    local_model_grads = get_gradients(global_m,
                                      local_models)  # 这个和上面的get_matrix_similarity_from_grads函数里面的local_model_grads是一样的

    metric_matrix = np.zeros((n_clients, n_clients))  # 定义个nclient维度的方阵
    for i, j in product(range(n_clients), range(n_clients)):
        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
            # 为什么我感觉实际内容和上面的函数get_matrix_similarity_from_grads是一样的？
        )

    return metric_matrix


def get_clusters_with_alg2(linkage_matrix: np.array, n_sampled: int, weights: np.array):  # 这个是根据算法2的聚类函数
    # 输入分别是linkage矩阵，抽样的个数，权重
    """Algorithm 2"""
    epsilon = int(10 ** 10)

    # associate each client to a cluster
    link_matrix_p = deepcopy(linkage_matrix)  # 这个linkage是python的层次聚类，linkage_matrix就是层次聚类矩阵
    ''''linkage函数从字面意思是链接，层次分析就是不断链接的过程，最终从n条数据，经过不断链接，最终聚合成一类，算法就此停止。'''
    augmented_weights = deepcopy(weights)  # 这个是通过deepcopy得到的增强权重数组（目前只是将权重数组拷贝一下）。（可以确认是权重向量数组了）

    for i in range(len(link_matrix_p)):  # 这种输出的应该是 层次聚类矩阵的行数，参考len的用法，a是矩阵，len(a)=行数，len(a[0])是列数
        idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])  # idx_1指的是层次聚类矩阵link_matrix_p的第1列的第n+1行
        # idx_2指的是link_matrix_p矩阵的第2列中的第i+1行。

        new_weight = np.array(  # 生成一个新数组，记作new_weight，里面的值是，augmented_weights矩阵中的第1列和第2列的第n+1行的数值相加
            [augmented_weights[idx_1] + augmented_weights[idx_2]]
        )
        augmented_weights = np.concatenate(
            (augmented_weights, new_weight))  # 将new_weights与augmented_weights合并起来，作为新的augmented_weights
        link_matrix_p[i, 2] = int(new_weight * epsilon)  # link_matrix_p矩阵第三列的第i+1行，将是new_weight * epsilon。

    clusters = fcluster(
        link_matrix_p, int(epsilon / n_sampled), criterion="distance"
    )  # 这个是numpy中fcluster函数的用法，1.参数Z是linkage函数的输出Z。2.参数scalar：形成扁平簇的阈值。3.参数criterion。fcluster(t,z,criterion)
    # 这个直接输出“从给定链接矩阵定义的层次聚类中形成平面聚类”

    n_clients, n_clusters = len(clusters), len(set(clusters))  # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    # 此时n_clients = len(clusters), n_clusters = len(set(clusters)).

    # Associate each cluster to its number of clients in the cluster
    pop_clusters = np.zeros((n_clusters, 2)).astype(int)  # 定义一个满是0的矩阵，行数是聚类的类别个数，两列。
    for i in range(n_clusters):  # 其实这个是相当于每一行进行一个遍历
        pop_clusters[i, 0] = i + 1  # 这个pop_clusters矩阵，第一列，从第一行到最后一行开始，分别是1，,2，,3....，n_clusters
        for client in np.where(clusters == i + 1)[
            0]:  # 这块分成两个步骤，首先使用np.where给出符合条件的索引，这个np.where结果第一行就是索引的行数，第二行就是索引的列数，此时我们只要行数，所以直接[0]
            pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)  # 这个就是pop_cluster的第二列，是计算关于权重的一些信息。

    pop_clusters = pop_clusters[
        pop_clusters[:, 1].argsort()]  # 首先对pop这个数组的第二列的每一行，进行从小到大的排序，但是argsort给出的是索引号，通过pop_clusters[索引号]便可得到对应的值
    # 其实就是根据pop_clusters的第二列进行从小到大进行排序。

    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)  # 定一个矩阵，维度分别是n_sampled和n_clients的

    # n_sampled biggest clusters that will remain unchanged
    kept_clusters = pop_clusters[n_clusters - n_sampled:, 0]  # 这个是取pop_clusters的第一列中从n_clusters - n_sampled以后的所有行的数据，
    # 如果n_cluster = 5, n_sampled = 3,那么就是取第2行以后的数据，哦哦，对了，这个pop_clusters是按照从小到大排列的，所以这个的确就是为了使得n_sampled个clusters保持不变，这也是说明了
    # n_sampled指的是那些biggest clusters的个数。

    for idx, cluster in enumerate(kept_clusters):  # enumerate函数是对于kept_clusters里面每一个元素，都对应加一个索引号
        for client in np.where(clusters == cluster)[0]:
            distri_clusters[idx, client] = int(
                weights[client] * n_sampled * epsilon
            )

    k = 0  # 定义一个变量k=0
    for j in pop_clusters[: n_clusters - n_sampled, 0]:  # 对于pop_clusters的第一列中从n_clusters - n_sampled以后的所有行的数据，进行遍历

        clients_in_j = np.where(clusters == j)[0]  # 这个clients_in_j应该是一个数组，数组里面的内容是第j个聚类中的客户端
        np.random.shuffle(clients_in_j)  # np.random.shuffle这个函数的作用就是为了打乱数据的顺序。

        for client in clients_in_j:

            weight_client = int(
                weights[client] * epsilon * n_sampled)  # 给第j个聚类中的参与方进行赋予权重，计算公式就是weights[client] * epsilon * n_sampled

            while weight_client > 0:  # 如果客户端的群众大于0

                sum_proba_in_k = np.sum(distri_clusters[k])

                u_i = min(epsilon - sum_proba_in_k, weight_client) # 这个应该是取这两个变量最小的那个作为u_i的值

                distri_clusters[k, client] = u_i
                weight_client += -u_i

                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters


from numpy.random import choice


def sample_clients(distri_clusters):
    n_clients = len(distri_clusters[0])
    n_sampled = len(distri_clusters)

    sampled_clients = np.zeros(len(distri_clusters), dtype=int)

    for k in range(n_sampled):
        sampled_clients[k] = int(choice(n_clients, 1, p=distri_clusters[k]))

    return sampled_clients
