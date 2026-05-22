# Standard PyTorch imports
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values




X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

# 2. 关键一步：把 NumPy 数组转换为 PyTorch Tensor，并指定数据类型（通常特征用 float32）
X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)