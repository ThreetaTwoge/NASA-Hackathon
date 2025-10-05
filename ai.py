import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from random import randint


def min_max_normalize(tensor, min_range=-1, max_range=1):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
    return normalized_tensor


database = pd.read_csv("exoTrain.csv")
array = database.to_numpy(dtype=np.float32)
tensor = torch.from_numpy(array)
filter1 = tensor[tensor[:, 0] == 2]
print(filter1)

'''
xs = min_max_normalize(torch.from_numpy(array[:, 1:]))
ys = torch.from_numpy(array[:, 0]) - 1


class ExoFinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(3197, 2000, bias=True)
        self.Matrix2 = nn.Linear(2000, 1000, bias=True)
        self.Matrix3 = nn.Linear(1000, 600, bias=True)
        self.Matrix4 = nn.Linear(600, 200, bias=True)
        self.Matrix5 = nn.Linear(200, 10, bias=True)
        self.Matrix6 = nn.Linear(10, 1, bias=True)
        self.R = nn.ReLU()
        self.S = F.tanh
    def forward(self, x):
        x = x.view(-1, 3197)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.R(self.Matrix3(x))
        x = self.R(self.Matrix4(x))
        x = self.R(self.Matrix5(x))
        x = self.R(self.Matrix6(x))
        x = self.S(x)
        return x.squeeze()

total = len(ys)
f = ExoFinder()
L = nn.BCELoss()
print(L(f(xs), ys))

optimizer = torch.optim.SGD(f.parameters(), lr=0.001, momentum=0.9)
f.train()

for _ in range(1000):
    k = randint(0, total - 1)
    y = f(xs[:k])
    loss = L(y, ys[:k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


database = pd.read_csv("exoTest.csv")
array = database.to_numpy(dtype=np.float32)
xs = min_max_normalize(torch.from_numpy(array[:, 1:]))
ys = torch.from_numpy(array[:, 0]) - 1

print(L(f(xs), ys))

PATH = "my_model.pth"
torch.save(f.state_dict(), PATH)
'''