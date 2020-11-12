from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class ProtConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, dropout=None):
        super(ProtConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_out,
                            kernel_size=kernel_size, padding=padding)

        self.act = nn.ReLU()
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, fea):
        if len(fea.size()) == 2:
            fea = fea.unsqueeze(1)

        x = self.conv(fea)

        x = self.act(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, training=self.training)

        return x
