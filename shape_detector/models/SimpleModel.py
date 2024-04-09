import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.l1 = nn.Linear(63, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.softmax(x, dim=0)
        return x