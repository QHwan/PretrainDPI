from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .concrete import ConcreteDropout

class GraphNet(nn.Module):
    def __init__(self, dim_node, dim_edge, dropout):
        super(GraphNet, self).__init__()
        self.fc_edge = nn.Linear(dim_node+dim_edge, dim_edge)
        self.fc_node = nn.Linear(dim_node+dim_edge, dim_node)

        self.act = nn.ReLU()
        self.dropout = dropout
        self.ConDrop = ConcreteDropout()

        torch.nn.init.xavier_normal_(self.fc_edge.weight)
        torch.nn.init.xavier_normal_(self.fc_node.weight)

    def forward(self, node, edge, e2n):
        #node->edge
        x_edge = torch.cat([edge, torch.mm(torch.transpose(e2n, 0, 1), node)], dim=-1)
        if self.dropout is not None:
            x_edge = self.ConDrop(x_edge, self.fc_edge)
        x_edge = self.act(x_edge)
        
        #edge->node
        x_node = torch.cat([node, torch.matmul(e2n, x_edge)], dim=-1)
        if self.dropout is not None:
            x_node = self.ConDrop(x_node, self.fc_node)
        x_node = self.act(x_node)
        
        x_node = x_node + node
        x_edge = x_edge + edge

        return(x_node, x_edge)


class GraphPool(nn.Module):
    def __init__(self, reduce):
        super(GraphPool, self).__init__()
        self.reduce = reduce

    def forward(self, fea, idx_fea):
        x = torch_scatter.scatter(fea, idx_fea, dim=0, reduce=self.reduce)
        return x
