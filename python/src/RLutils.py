import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, h1=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_neural_network():
    model = torch.zeros([2, 4], dtype=torch.int32) #MLP()
    return model