# coding: utf-8
import torch
import pickle
import sys
import numpy as np
sys.path.append("..")
from py_func.create_model import NN
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


#################  加载模型  #################
model = NN(50, 10)

state_dict = torch.load("../saved_exp_info/final_model/MNIST_shard_random_any_i600_N50_lr0.01_B50_d1.0_p0.1_m2_0.pth")

model.load_state_dict(state_dict)

model.eval()

#################  加载测试集  #################
_, list_mnist_test = get_dataloaders("MNIST_shard", 50) 


# mnist_test = MNIST(
#             root="./data",
#             train=False,
#             download=True,
#             transform=transforms.ToTensor(),
#         )


for i in range(1000):
	new_indices = np.random.choice(len(mnist_test), size=len(mnist_test), replace=True)
	sampled_trainset = [mnist_test[i] for i in new_indices]
	test_loader = DataLoader(sampled_trainset, batch_size=50, shuffle=True)

	correct = 0
	for features, labels in test_loader:
		predictions = model(features) # 对应注释中，我们根据feature和model，来得出label的prediction
		_, predicted = predictions.max(1, keepdim=True)  # 我明白了，因为.max函数输出的是两个，第一个是每一行的最大值，第二个是每一行中最大值所在的位置，而我们需要的是位置吧，所以前面的“_”仅仅是用于占位的。我们需要的是每一行中最大值所在的位置的索引号  
		# 这个Keepdim=True仅仅是为了维持其二维特性，即对应位置操作。
		# print(predicted.view(-1, 1))
		# print(labels.view(-1, 1))
		correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()  # 首先，这个view(-1,1) 是分别让predicted和view这两个array按每一行从左到右的顺序，按列进行排列，然后两列数据，进行个对比。计算出相等的个数，然后加起来。
		# print(correct)
	accuracy = 100 * correct / len(test_loader.dataset)  # 计算模型在测试数据集上的准确度。
	# print(f"====> Accuracy: {accuracy}")
	print(accuracy)