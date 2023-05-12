#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

'''
要使用torch.optim，必须构造一个优化器对象，该对象将保存当前状态并根据计算出的梯度更新参数。
要构造一个优化器，必须给它一个包含要优化的参数的迭代，制定特定于优化器的选项，例如学习率、权重衰减等
'''

import numpy as np
from copy import deepcopy

import random # 用于产生随机数

def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data) # layer_weights = layer_weights - layer_weights，然后layer_weights=0

# FedAvg的聚合过程
"""
这个clients_models_hist指的是历史训练数据，一般Model.hist的右式是model.fit()函数，这个函数的返回值是返回一个History的对象，
这个history记录了损失函数和其他指标的数值随epoch变化的情况。
"""
def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):  # 输入就是模型，客户端模型的历史数值，以及权重值。
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model) # 首先复制一个model过来，当做一个容器（任意模型）
    set_to_zero_model_weights(new_model)  # 我们将这个模型通过递减的形式将所有参数变成0，就变成一个全新的model容器了。

    for k, client_hist in enumerate(clients_models_hist): # enumerate函数用于for循环，可以同时获得索引和值，即index和value。这个k和client_hist应该就是从enumerate(clients_models_hist)中得到的

        for idx, layer_weights in enumerate(new_model.parameters()): # 这个idx和layer_weights是从enumerate(new_model.parameters())得到的

            contribution = client_hist[idx].data * weights[k] # 这个应该是计算每个客户端的贡献值，其实也就是客户端自己的模型参数乘以对应的权重。
            layer_weights.data.add_(contribution) # 这个应该是递增 layer_weights.data = layer_weights.data + contribution,其实这个就是全局模型的权重值

    return new_model


def FedAvg_agregation_process_for_FA_sampling(
    model, clients_models_hist: list, weights: list
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)

    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights) * layer_weigths.data) # 这个函数主要是和上面的set_to_zero_model_weights函数的内容不同。
                    # layer_weigths = layer_weigths - sum(weights) * layer_weigths，但是这样的话，模型应该无法清空吧，无法当一个纯净的容器。
    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model

#  输入分别是模型和数据集，对于每一种模型，只要调用这个accuracy_dataset便可以计算准确率。
def accuracy_dataset(model, dataset):  # 计算在测试数据集上模型的准确度
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0 # 应该是先定义一个变量correct=0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    机器学习中有label和feature两个概念：
    1. label是分类，是我们要预测的东西。 
    2. feature是特征，应该就是统计中的指标，比如 黄色，圆
    3. 如果训练出feature和label之间的关系，那我们可以通过feature来预测label的值。
    ''' 
    for features, labels in dataset:
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features) # 对应注释中，我们根据feature和model，来得出label的prediction
        _, predicted = predictions.max(1, keepdim=True)  # 我明白了，因为.max函数输出的是两个，第一个是每一行的最大值，第二个是每一行中最大值所在的位置，而我们需要的是位置吧，所以前面的“_”仅仅是用于占位的。我们需要的是每一行中最大值所在的位置的索引号
            # 这个Keepdim=True仅仅是为了维持其二维特性，即对应位置操作。
        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()  # 首先，这个view(-1,1) 是分别让predicted和view这两个array按每一行从左到右的顺序，按列进行排列，然后两列数据，进行个对比。计算出相等的个数，然后加起来。

    accuracy = 100 * correct / len(dataset.dataset)  # 计算模型在测试数据集上的准确度。

    return accuracy

#  输入分别是模型和数据集以及计算loss的函数loss_f，对于每一种模型，只要调用这个loss_dataset便可以计算在某个数据集上的loss.
def loss_dataset(model, train_data, loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for idx, (features, labels) in enumerate(train_data): # 首先给数据集的每一个数据加一个索引号
        features = features.to(device)   
        labels = labels.to(device) 
        predictions = model(features)
        loss += loss_f(predictions, labels) # 嵌入loss_f函数，输入预测值与label值，计算损失。

    loss /= idx + 1
    return loss

# 计算交叉熵损失函数
def loss_classifier(predictions, labels):
    # zzy begin
    labels = labels.long()
    # zzy end
    criterion = nn.CrossEntropyLoss()  # 这个是直接调用了pytorch里面的 交叉熵损失函数，用于解决多分类问题。
    return criterion(predictions, labels)

# 计算模型中参数的个数
def n_params(model):
    """return the number of parameters in the model"""

    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])  # np.prod函数默认是计算所有元素的乘积。
            for tensor in list(model.parameters()) # 对于模型参数的每一个tensor中
        ]   # 我们对每个tensor内部，对每个数据进行先遍历得到所有数据后，然后乘积操作。
    ) # 需要整体的计算求和。

    return n_params

# 计算两个模型的参数之间的2范数差值
def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2) # 这个就是计算二范数，即MSE，差值的平方和
            for i in range(len(tensor_1))
        ]
    )  # 需要总体的进行求和。

    return norm

# 本地进行学习的函数。
def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):
# 输入的分别是模型、mu、优化器、训练数据集、SGD的次数、计算loss的函数loss_f
    model_0 = deepcopy(model)  # 首先把输入的模型给deepcopy一下，放到本地记作model_0，用于后面的对比。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for _ in range(n_SGD):
        '''
        这个iter是一个迭代器，这个next是返回迭代器的下一个项目。
        但是其实这个next(iter())就是将train data里面的每一个数据分别迭代return出来。
        '''
        features, labels = next(iter(train_data)) # 可能是这个train_data里面就是包含了feature和label。
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad() # 这句的意思是把梯度初始化零，也就是把loss关于weight的导数变为0。基本上对每个batch都执行了这个操作。

        predictions = model(features) # 然后基于model和feature，来得到label的预测值

        batch_loss = loss_f(predictions, labels) # 计算预测与label之间的损失，这个是一个batch中产生的损失
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0) # 计算模型与刚开始传进来的model_0进行参数上的比较，计算2范数的difference，累加到batch_loss中

        batch_loss.backward() # 反向传播计算得到每个参数的梯度值
        optimizer.step() # 通过梯度下降执行一步参数更新
        # optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。


# 这个pickle可以将对象以文件的形式存放在磁盘上。
import pickle

def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output) # pickle.dump函数可以序列化对象，并将结果数据流写入到文件对象中。

# 这个函数输入客户端和服务器中具有common structure的model,n_sampled，训练集，测试集，迭代的次数，SGD的次数，优化器的学习率，文件名，decay，正则项mu等。
# 输出的是最终的Global model
# 这个按照文中所示，应该是联邦学习基于MD抽样的函数？还是只是 随机抽样的 联邦学习函数
def FedProx_sampling_random(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    '''
    这个metric_period=1的意思是：
    外层循环多少次，计算一次metric.比如，metric_period=2，那么当i=0, 2, 4, 6....的时候计算一次
    不过默认是1。
    '''
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier  # 这个属于是调用了loss_classifier函数，其实就是调用交叉熵损失函数

    K = len(training_sets)  # number of clients 记K为测试集的长度
    n_samples = np.array([len(db.dataset) for db in training_sets]) # n_samples是一个数组。
    weights = n_samples / np.sum(n_samples) # 计算权重值，根据数量占总体多少来计算
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))  # 定义一个array，维度分别是n_iter+1, K,记作loss_hist，应该是用于盛放loss的历史值
    acc_hist = np.zeros((n_iter + 1, K))   # 定义一个array，维度分别是n_iter+1, K,记作acc_hist，应该是用于盛放准确率的历史值

    for k, dl in enumerate(training_sets): # 将训练集数据enumerate一下，k是索引号，dl是value

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach()) # 计算前K个数据的loss，“调用了loss_dataset函数”，然后结果用detach函数变成np.array，方便进一步操作
        acc_hist[0, k] = accuracy_dataset(model, dl) # 调用accuracy_dataset函数，计算前K个数据的精确度

    # LOSS AND ACCURACY OF THE INITIAL MODEL 初始模型的loss和精确度
    server_loss = np.dot(weights, loss_hist[0]) # .dot函数是计算计算两个矩阵或者向量的乘积的。这个是将权重与前K个数据的loss进行相乘，然后进行个求和，作为服务器的loss
    server_acc = np.dot(weights, acc_hist[0]) # 这个将权重与前K个数据的acc进行相乘，作为服务器精确率
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}") # 这个是运行程序时，每迭代一次，输出一次。print里面用f""其实可以使用花括号里面的变量和表达式。

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int) # 这个应该是将n轮迭代中，每一轮抽样出来的K个客户端的history，这个是先创建一个容器占位。
    # 这个应该是具体的每一轮迭代中，如何将抽样出的K个参与方放进sampled_clients_hist中
    for i in range(n_iter):

        clients_params = []  # 创建一个容器，用于放来自客户端的params

        np.random.seed(i) # 保证每次产生的随机数都是可以复现
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        ) # 这个是从数组中随机抽取数据，是放回抽样

        for k in sampled_clients:    # 在随机抽样出的K个参与方中，做一个循环

            local_model = deepcopy(model) # 将输入进来的model，深度拷贝一下作为local model
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr) # 然后在local model上的参数运行SGD优化器，lr指的是学习率，也是函数的一个输入对象

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )  # 调用local_learning函数，将参数输入进去

            # GET THE PARAMETER TENSORS OF THE MODEL  # 应该是得到local model的参数的tensors
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params] # 将list_params里面的每一个param的tensor挨个进行detach，再重新放入list_params
            clients_params.append(list_params) # 将model中的参数的tensor解出来，放进之前创建好的clents_params列表中

            sampled_clients_hist[i, k] = 1  # 这个应该是对于每一轮，谁被抽到了，就在sampled_clients_hist这个数组中标记一个位置。

        # CREATE THE NEW GLOBAL MODEL  这个是创建每一轮迭代中的全局模型
        model = FedAvg_agregation_process(  # 调用了FedAvg的聚合过程
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        ) # 输入一些参数，分别是模型、客户端的参数、权重

        if i % metric_period == 0: # 因为metric_period是1，所以每个i都会进行计算
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL 计算不同的客户端对于新模型的服务器loss和服务器acc
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(  # 因为这个i是迭代嘛，所以这个loss_hist的比如 第一行是client的loss，则第二行（i+1）是服务器的loss
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl) # 同上loss的情况。

            server_loss = np.dot(weights, loss_hist[i + 1]) # 根据weights和第二行（i+1）的sever的loss进行.dot乘积求和
            server_acc = np.dot(weights, acc_hist[i + 1]) # 同上sever_loss

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay # 学习率随着训练进行有效地降低，保证收敛稳定性。

    # SAVE THE DIFFERENT TRAINING HISTORY  保存不同的训练history
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name) # 将loss_hist写进去pkl文件
    save_pkl(acc_hist, "acc", file_name) # 将acc_hist写进去pkl文件
    save_pkl(sampled_clients_hist, "sampled_clients", file_name) # 将每一轮迭代中，抽样的客户端的编号的hist写进去pkl文件

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )  # 只保存训练好的权重，torch.save(model.state_dict()). 如果是保存整个模型就是torch.save(model)

    return model, loss_hist, acc_hist # 返回模型(这个是全局模型)、loss_hist、acc_hist

# 这个是基于聚类抽样的联邦学习过程，整体应该与上个模型类似，但是中间的抽样算法会有改变，最终返回的是基于聚类抽样的联邦学习全局模型
def FedProx_clustered_sampling(
    sampling: str, # 这个random指的是The `sampling` scheme used. Either `random` for MD sampling, `clustered_1` and `clustered_2` for clustered sampling with Algorithm 1 and 2, or `FedAvg` for the initial sampling scheme proposed in FedAvg.
    model,
    n_sampled: int,
    training_sets: list, #  list of the training sets. At each index is the training set of client "index"
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    sim_type: str, # 是计算相似度的方法类型。The similarity measure `sim_type` used for the clients representative gradients. With `clustered_2` put either `cosine`, `L2` or `L1` and, with other sampling, put `any`.
    iter_FP=0,
    decay=1.0, # 学习率递减所需的一个参数
    metric_period=1,
    mu=0.0, # The local loss function regularization parameter `mu`. Leaving this field empty gives no local regularization, `mu=0`
):
    """all the clients are considered in this implementation of FedProx  考虑了所有的参与方
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    from scipy.cluster.hierarchy import linkage # 这个是层次聚类的函数包
    from py_func.clustering import get_matrix_similarity_from_grads # 导入计算梯度矩阵相似度的函数，因为这个调用到clustering.py里面的函数，所以需要这样，先写一个函数头。
    if sampling == "clustered_2":  # 如果抽样方式是clustered_2，则采用get_clusters_with_alg2函数。
        from py_func.clustering import get_clusters_with_alg2
    from py_func.clustering import sample_clients # 从clustering.py函数里面调用sample_clients函数

    loss_f = loss_classifier  # 虽然loss_classifier函数在本py文件中，但是如果在函数内调用，仍然需要写一遍。记作计算loss的函数

    # Variables initialization 变量初始化
    K = len(training_sets)  # number of clients 记K为训练集的长度，也是clients的数量。list of the training sets. At each index is the training set of client "index"
    n_samples = np.array([len(db.dataset) for db in training_sets]) # n_samples首先是个数组，里面的每一个数据，应该是训练集中每一个客户端的样本数或者说数据量。
    weights = n_samples / np.sum(n_samples) # 毕竟权重就是根据每个抽样出的客户端的数据量占总数据量的比例。
    print("Clients' weights:", weights) # 输出 “客户端权重”

    loss_hist = np.zeros((n_iter + 1, K)) # 先创建一个loss的history矩阵，分别是n_iter+1行，k列
    acc_hist = np.zeros((n_iter + 1, K)) # 创建acc的history矩阵，分别是n_iter+1行，k列

    for k, dl in enumerate(training_sets): # 将训练集进行enumerate一下，k是索引号，dl是value
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach()) # 计算前K个数据的loss，“调用了loss_dataset函数”，然后结果用detach函数变成np.array，并变成浮点型。其实就是loss_hist的第一行数据
        acc_hist[0, k] = accuracy_dataset(model, dl) # 计算前k个数据的acc。（这个[0,k]指的是第一行，第k-1个的那个数，其实就是acc_hist第一行的数据）

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0]) # 计算服务器的loss，直接就是矩阵的相乘np.dot，权重乘以loss_hist
    server_acc = np.dot(weights, acc_hist[0]) # 计算服务器的acc，权重乘以acc_hist
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}") # 这个是在运行程序的时候输出的东西

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int) # 创建个矩阵，用于放每次抽样时抽出来的客户端，n_iter行 k列

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0 将梯度的所有值都初始化为一列0

    if sampling == "clustered_1": # 如果抽样方式等于"clustered_1"
        from py_func.clustering import get_clusters_with_alg1  # 从另一个py文件中clustering.py调用get_clusters_with_alg1函数

        distri_clusters = get_clusters_with_alg1(n_sampled, weights)

    elif sampling == "clustered_2":
        from py_func.clustering import get_gradients

        gradients = get_gradients(sampling, model, [model] * K) # 因为算法2是通过计算梯度相似度来进行抽样的

    for i in range(n_iter):

        previous_global_model = deepcopy(model) # deepcopy一下输入的模型作为最初的全局模型

        clients_params = [] # 创建个列表，放clients的参数
        clients_models = [] # 创建个列表，放clients的模型
        sampled_clients_for_grad = [] # 感觉这个应该是创建个列表，用于盛放 为了算法2，依据相似度进行聚类的，我们要找出梯度。

        if i < iter_FP: # 这个iter_FP 在输入进函数时，是iter_FP = 0
            print("MD sampling")

            np.random.seed(i) # 设置seed
            sampled_clients = np.random.choice(
                K, size=n_sampled, replace=True, p=weights
            ) # 随机放回抽样出客户端，抽样的概率是各自的权重。

            for k in sampled_clients: # 对于抽样出来的客户端中

                local_model = deepcopy(model) # 通过deepcopy一下输入寄哪里的model来作为local model
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr) # 本地模型参数上运行SGD优化器

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                ) # 调用local_learning函数，将一些参数都输入进去。

                # SAVE THE LOCAL MODEL TRAINED 保存训练好的本地模型
                list_params = list(local_model.parameters()) # 定义一个变量list_params，是本地模型的参数列表
                list_params = [
                    tens_param.detach() for tens_param in list_params # 将list_params里面的每一个param的tensor挨个进行detach，再重新放入list_params
                ] #
                clients_params.append(list_params) # 将model中的参数的tensor解出来，放进之前创建好的clents_params列表中
                clients_models.append(deepcopy(local_model)) # deepcopy一下本地模型，然后放进之前创建好的client_models里面

                sampled_clients_for_grad.append(k)  # 这个就是第k个参与方进行训练后，就把这第k个参与方放进去sampled_clients_for_grad这个列表中
                sampled_clients_hist[i, k] = 1 # 这个应该是做一个标记，第k个进行训练后，就标记个1


        else:
            if sampling == "clustered_2": # else 如果抽样方式是 cluster_2，就是基于客户相似性矩阵进行聚类的

                # GET THE CLIENTS' SIMILARITY MATRIX 得到客户端的相似性矩阵
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                ) # 计算矩阵相似度，通过调用get_matrix_similarity_from_grads函数

                # GET THE DENDROGRAM TREE ASSOCIATED
                linkage_matrix = linkage(sim_matrix, "ward") # 通过linkage()函数来得到层次分析矩阵，linkage直接输出的就是一个矩阵，告诉你谁和谁的距离是多少，是否可以合并，合并后有几个元素；这个ward指的是离差平方和距离

                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, n_sampled, weights
                ) # 调用get_clusters_with_alg2得到基于矩阵相似度的聚类结果

            for k in sample_clients(distri_clusters): # 从distri_cluster进行客户端抽样，然后在抽出来的客户端中，进行for循环以下操作

                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                ) # 和前面MD抽样一样，依旧是deepcopy输入的model，导入SGD优化器等等训练必须的参数

                # SAVE THE LOCAL MODEL TRAINED 保存训练好的本地模型
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        ) # 调用fedavg聚合函数来获取全局模型

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        if i % metric_period == 0:

            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # UPDATE THE HISTORY OF LATEST GRADIENT
        if sampling == "clustered_2":
            gradients_i = get_gradients(
                sampling, previous_global_model, clients_models
            ) # 直接调用get_gradients函数，分别输入刚开始的全局模型，以及这一轮迭代中的本地模型们，计算"""return the `representative gradient` formed by the difference between
              # the local work and the sent global model"""  # 通过对比本地工作与全局模型之间的差异来得出代表性梯度
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i): # zip函数将两组数据依照位置对应起来形成元组，整体按照列表排列
                gradients[idx] = gradient

        lr *= decay # 学习率递减的一个表达式

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    ) # 保存一个序列化目标，只能通过python打开

    return model, loss_hist, acc_hist

#
def FedProx_sampling_target(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    # Variables initialization
    n_samples = sum([len(db.dataset) for db in training_sets])
    weights = [len(db.dataset) / n_samples for db in training_sets]
    print("Clients' weights:", weights)

    loss_hist = [
        [
            float(loss_dataset(model, dl, loss_f).detach()) # dl 是数据集的值
            for dl in training_sets
        ]
    ]
    acc_hist = [[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist = [
        [tens_param.detach().numpy() for tens_param in list(model.parameters())]
    ]
    models_hist = []
    sampled_clients_hist = []

    server_loss = sum(
        [weights[i] * loss_hist[-1][i] for i in range(len(weights))] # 这个应该指的是乘以loss_hist的最后一行的第i+1列
    )
    server_acc = sum(
        [weights[i] * acc_hist[-1][i] for i in range(len(weights))] # 乘以 acc_hist的最后一行的第i+1列
    )
    print(f"====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}")

    for i in range(n_iter):

        clients_params = []
        clients_models = []
        sampled_clients_i = []

        for j in range(n_sampled):

            k = j * 10 + np.random.randint(10) # np.random.randint函数的作用是返回一个随机整型数，由于没有标明high的值，所以默认是返回[0,10)的值

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters()) # 得到本地训练的模型参数
            list_params = [tens_param.detach() for tens_param in list_params] # 挨个进行detach
            clients_params.append(list_params) # 将list_params加入到clients_params这个列表或者矩阵中
            clients_models.append(deepcopy(local_model)) # 将local_model加入进clients_models列表中

            sampled_clients_i.append(k)

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )
        models_hist.append(clients_models)

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [
            [
                float(loss_dataset(model, dl, loss_f).detach())
                for dl in training_sets
            ]
        ]
        acc_hist += [[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss = sum(
            [weights[i] * loss_hist[-1][i] for i in range(len(weights))]
        )
        server_acc = sum(
            [weights[i] * acc_hist[-1][i] for i in range(len(weights))]
        )

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        server_hist.append(deepcopy(model))

        sampled_clients_hist.append(sampled_clients_i)

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

# FedAvg
def FedProx_FedAvg_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)
        print("sampled clients", sampled_clients)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process_for_FA_sampling(
            deepcopy(model),
            clients_params,
            weights=[weights[client] for client in sampled_clients],
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist
