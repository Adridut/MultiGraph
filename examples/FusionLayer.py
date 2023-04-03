import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):

    def __init__(self, n_output, w):
        super(FusionLayer, self).__init__()
        self.W = nn.Parameter(torch.tensor(w))

    def forward(self, inputs, w):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [i.unsqueeze(-1) for i in inputs]
        inputs = torch.cat(inputs, dim=-1) # (n_batch, n_feat, n_inputs)
        weights = torch.softmax(self.W, dim=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return torch.sum(weights*inputs, dim=-1), weights # (n_batch, n_feat)
