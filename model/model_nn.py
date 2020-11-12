from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .model_drug import GraphNet, GraphPool
from .model_prot import ProtConv

def create_model(args):
    model = GNetNN(args)
    return model


class Perceptron(nn.Module):
    def __init__(self, dim_in, dim_out, act=None, dropout=None):
        super(Perceptron, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)

        self.act = act
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)

        if self.act == 'relu':
            x = torch.relu(x)
        elif self.act == 'log_softmax':
            x = F.log_softmax(x, 1)
        elif self.act == 'sigmoid':
            x = torch.sigmoid(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, training=self.training)

        return x


class DrugEncoder(nn.Module):
    def __init__(self, args):
        super(DrugEncoder, self).__init__()

        self.embed_node = Perceptron(args.dim_drug_node, args.dim_enc_drug)
        self.embed_edge = Perceptron(args.dim_drug_edge, args.dim_enc_drug)

        self.gl_list = nn.ModuleList([])
        for i in range(args.n_layer_drug):
            if i == args.n_layer_drug - 1:
                gl = GraphNet(args.dim_enc_drug, args.dim_enc_drug, dropout=args.dropout)
            else:
                gl = GraphNet(args.dim_enc_drug, args.dim_enc_drug, dropout=None)
            self.gl_list.append(gl)

        self.pool_node = GraphPool(reduce='mean')
        self.pool_edge = GraphPool(reduce='mean')

        self.fc = Perceptron(args.dim_enc_drug+args.dim_enc_drug, args.dim_enc_drug, act='relu')

    def forward(self, drug):
        node, edge, n2n, e2n, idx_node, idx_edge = drug

        x_node = self.embed_node(node)
        x_edge = self.embed_edge(edge)

        for gl in self.gl_list:
            x_node, x_edge = gl(x_node, x_edge, e2n)

        x_node = self.pool_node(x_node, idx_node)
        x_edge = self.pool_edge(x_edge, idx_edge)

        x = torch.cat([x_node, x_edge], -1)

        x = self.fc(x)

        return x_node



class ProtEncoder(nn.Module):
    def __init__(self, args):
        super(ProtEncoder, self).__init__()
        self.prot_fc1 = ProtConv(1, 16, 8, 0)
        self.prot_fc2 = ProtConv(16, 32, 8, 0)
        self.prot_fc3 = ProtConv(32, 48, 8, 0, dropout=args.dropout)

        self.fc = Perceptron(args.dim_prot-21, args.dim_enc_prot, act='relu')

    def forward(self, prot):
        x = prot

        x = self.prot_fc1(x)
        x = self.prot_fc2(x)
        x = self.prot_fc3(x)

        x = torch.max(x, 1)[0]

        x = self.fc(x)

        return x



class GNetNN(nn.Module):
    def __init__(self, args):
        super(GNetNN, self).__init__()

        self.drug_encoder = DrugEncoder(args)
        self.prot_encoder = ProtEncoder(args)

        self.mlp_list = nn.ModuleList([])
        for i in range(args.n_layer_mlp):
            if i == 0:
                fc = Perceptron(args.dim_enc_drug + args.dim_enc_prot, args.dim_mlp,
                                act='relu', dropout=args.dropout)
            elif i == args.n_layer_mlp - 1:
                fc = Perceptron(args.dim_mlp, 1, act='sigmoid')
            else:
                fc = Perceptron(args.dim_mlp, args.dim_mlp,
                               act='relu', dropout=args.dropout)
            self.mlp_list.append(fc)

    def forward(self, prot, drug):
        x_drug = self.drug_encoder(drug)
        x_prot = self.prot_encoder(prot)

        x = torch.cat([x_prot, x_drug], -1)

        for mlp in self.mlp_list:
            x = mlp(x)

        return x