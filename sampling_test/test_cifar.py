# coding: utf-8
import torch
import pickle
import sys
import numpy as np
sys.path.append("..")
from py_func.FedProx import accuracy_dataset
from py_func.read_db import get_dataloaders
from py_func.create_model import CNN_CIFAR_dropout
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


#################  加载模型  #################
model = CNN_CIFAR_dropout()

state_dict = torch.load("../saved_exp_info/final_model/CIFAR10_nbal_0.01_clustered_1_any_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pth")

model.load_state_dict(state_dict)

model.eval()

#################  加载测试集  #################
# _, list_cifar_test = get_dataloaders("CIFAR10_nbal_0.01", 50) 

cifar_test = CIFAR10(
	root="./data",
	train=False,
	download=True,
	transform=transforms.ToTensor(),
)

for i in range(200):
	new_indices = np.random.choice(len(list_cifar_test), size=len(list_cifar_test), replace=True)
	list_sampled_test = [list_cifar_test[i] for i in new_indices]
	test_loader = DataLoader(list_sampled_test, batch_size=50, shuffle=True)
	acc_hist = np.zeros(100)
	for k, dl in enumerate(list_sampled_test):
		acc_hist[k] = accuracy_dataset(model, dl)
	server_acc = np.mean(acc_hist)
	print(server_acc)