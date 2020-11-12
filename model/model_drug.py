from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class GraphNet(nn.Module):
    def __init__(self, dim_node, dim_edge, norm=None, dropout=None):
        super(GraphNet, self).__init__()
        self.fc_edge = nn.Linear(dim_node+dim_edge, dim_edge)
        self.fc_node = nn.Linear(dim_node+dim_edge, dim_node)

        self.bn_edge = nn.BatchNorm1d(dim_edge)
        self.bn_node = nn.BatchNorm1d(dim_node)

        self.act = nn.ReLU()
        self.norm = norm
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.fc_edge.weight)
        torch.nn.init.xavier_uniform_(self.fc_node.weight)

    def forward(self, node, edge, e2n):
        #node->edge
        x_edge = torch.cat([edge, torch.mm(torch.transpose(e2n, 0, 1), node)], dim=-1)
        x_edge = self.fc_edge(x_edge)
        x_edge = self.act(x_edge)
        if self.dropout is not None:
            x_edge = F.dropout(x_edge, self.dropout, training=self.training)

        #edge->node
        x_node = torch.cat([node, torch.matmul(e2n, x_edge)], dim=-1)
        x_node = self.fc_node(x_node)
        x_node = self.act(x_node)
        if self.dropout is not None:
            x_node = F.dropout(x_node, self.dropout, training=self.training)

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
