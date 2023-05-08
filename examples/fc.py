import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x)
        return x