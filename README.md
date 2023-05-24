## prepare
先在项目根目录通过如下的四个命令跑mnist和cifar10的iid与non-iid的模型
```
python FL.py MNIST_iid random any 0 50 0.01 1.0 0.1 False 
python FL.py MNIST_shard random any 0 50 0.01 1.0 0.1 False 
python FL.py CIFAR10_iid random any 0 100 0.05 1.0 0.1 False
python FL.py CIFAR10_bbal_0.001 random any 0 100 0.05 1.0 0.1 False
```

然后进入sampling_test目录，分别运行如下4个命令跑跑mnist和cifar10的iid与non-iid的测试结果
```
python test.py mnist True
python test.py mnist False
python test.py cifar True
python test.py cifar False
```
最终测试结果会存储在目录sampling_test/saved_exp_info中