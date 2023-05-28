import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNConv
from dhg.structure import Hypergraph


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_conv: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        he_dropout: float = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
                HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate, he_dropout=he_dropout)
        )
        for i in range(num_conv-2):
            self.layers.append(
                HGNNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate, he_dropout=he_dropout)
            )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True, he_dropout=he_dropout)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)

        X = torch.sigmoid(X)

        return X
