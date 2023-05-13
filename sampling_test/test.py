# coding: utf-8
import torch
import pickle
import sys
import numpy as np
sys.path.append("..")
from sampling_test.read_db import get_dataloaders
from py_func.FedProx import accuracy_dataset
from py_func.create_model import NN
from py_func.create_model import CNN_CIFAR_dropout
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Experiment running on device: {}".format(device))
#################  加载模型  #################
dataset = sys.argv[1]
isiid = sys.argv[2] == "True"
if dataset == "mnist" :
	model = NN(50, 10)
	if isiid:
		state_dict = torch.load("../saved_exp_info/final_model/MNIST_iid_random_any_i600_N50_lr0.01_B50_d1.0_p0.1_m2_0.pth")
	else:
		state_dict = torch.load("../saved_exp_info/final_model/MNIST_shard_random_any_i600_N50_lr0.01_B50_d1.0_p0.1_m2_0.pth")
elif dataset == "cifar":
	model = CNN_CIFAR_dropout()
	if isiid:
		state_dict = torch.load("../saved_exp_info/final_model/CIFAR10_iid_random_any_i1000_N100_lr0.05_B200_d1.0_p0.1_m5_0.pth")
	else:
		state_dict = torch.load("../saved_exp_info/final_model/CIFAR10_nbal_0.001_random_any_i1000_N100_lr0.05_B200_d1.0_p0.1_m5_0.pth")

model.load_state_dict(state_dict)

model.to(device=device)

model.eval()


#################  加载测试集，打开文件  #################

if dataset == "mnist" :
	if isiid:
		file = open("saved_exp_info/MNIST_iid.txt", "w")
		file2 = open("saved_exp_info/MNIST_iid_python.txt", "w")
		list_ds_test = get_dataloaders("MNIST_iid", 50)
	else:
		file = open("saved_exp_info/MNIST_shard.txt", "w")
		file2 = open("saved_exp_info/MNIST_shard_python.txt", "w")
		list_ds_test = get_dataloaders("MNIST_shard", 50)
elif dataset == "cifar":
	if isiid:
		file = open("saved_exp_info/CIFAR10_iid.txt", "w")
		file2 = open("saved_exp_info/CIFAR10_iid_python.txt", "w")
		list_ds_test = get_dataloaders("CIFAR10_iid", 50)
	else:
		file = open("saved_exp_info/CIFAR10_nbal_0.001.txt", "w")
		file2 = open("saved_exp_info/CIFAR10_nbal_0.001_python.txt", "w")
		list_ds_test = get_dataloaders("CIFAR10_nbal_0.001", 50)


file2.write("datasets = [\n")
for (i, ds) in enumerate(list_ds_test):
	file2.write("[")
	file.write(f"client {i}\n")
	print(f"client {i}")
	for i in range(10):
		new_indices = np.random.choice(len(ds), size=len(ds), replace=True)
		sampled_ds = [ds[i] for i in new_indices]
		dl = DataLoader(sampled_ds, batch_size=200, shuffle=True)
		acc = accuracy_dataset(model, dl)
		file.write(f"{acc}\n")
		print(acc)
		file2.write(f"{acc}, ")
	file2.write("],\n")
file2.write("]\n")