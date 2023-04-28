import torch
import torch.nn as nn
import torch.optim as optim

# Daten
X = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.float32)
y = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)

# Modell definieren
class FC(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = nn.functional.softmax(self.fc(x), dim=1)
        return x