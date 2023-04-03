from dhg.models import HGNN
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHGNN(nn.Module):
    def __init__(self, nfeats, nhid, nclass, use_bn=False):
        super().__init__()
        self.hgnns = nn.ModuleList((
            HGNN(nfeat.shape[1], nhid, nclass, use_bn)
            for nfeat in nfeats
        ))
        self.dense_layer = nn.Linear(nclass, nclass)


    def forward(self, Xs, Hs):
        preds = torch.stack([ 
            hgnn(X, H) for hgnn, X, H in zip(self.hgnns, Xs, Hs)
        ])
        preds = preds.mean(0) 
        # preds = self.dense_layer(preds)
        preds =  F.softmax(preds, dim=1)
        return preds