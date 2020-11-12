from __future__ import print_function, division

import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score

import batch
import model.model_nn as model

def cal_roc_auc(label, pred):
    prob = pred
    score = roc_auc_score(label, prob)
    return score



def split(data_list, n_split, shuffle):
    if shuffle:
        np.random.shuffle(data_list)

    n_data = len(data_list)
    n_data_split = int(n_data / n_split) + 1

    split_list = []
    for i in range(n_split):
        if i == n_split - 1:
            split_list.append(data_list[i * n_data_split : ])
        else:
            split_list.append(data_list[i * n_data_split : (i+1) * n_data_split])

    return split_list



def train(args, data, model, criterion, optimizer):
    [prot_fea_list, drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list, label_list] = data

    prot_data = [prot_fea_list]
    drug_data = [drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list]
    label_data = [label_list]

    n_data = len(label_list)
    n_step = int(n_data / args.n_batch) + 1

    batch_idx_list = split(list(range(n_data)), n_step, shuffle=True)
    
    total_loss = 0
    model.train()
    for i, batch_idx in enumerate(batch_idx_list):
        batch_data = batch.make_batch(args, data=[prot_data, drug_data, label_data], idx=batch_idx)
        batch_prot_data, batch_drug_data, batch_label_data = batch_data

        optimizer.zero_grad()

        pred = model(batch_prot_data, batch_drug_data) 
        loss = criterion(pred.squeeze(), batch_label_data)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    return total_loss / n_step


def test(args, data, model, criterion, val):
    if val:
        shuffle=True
    else:
        shuffle=False
    [prot_fea_list, drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list, label_list] = data
    prot_data = [prot_fea_list]
    drug_data = [drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list]
    label_data = [label_list]

    n_data = len(label_list)
    n_step = int(n_data / args.n_batch) + 1

    batch_idx_list = split(list(range(n_data)), n_step, shuffle=shuffle)
    
    y_pred_list = []
    label_list = []
    total_loss = 0

    model.eval()
    for i, batch_idx in enumerate(batch_idx_list):
        batch_data = batch.make_batch(args, data=[prot_data, drug_data, label_data], idx=batch_idx)
        batch_prot_data, batch_drug_data, batch_label_data = batch_data

        y_pred = model(batch_prot_data, batch_drug_data)       
        loss = criterion(y_pred.squeeze(), batch_label_data)

        total_loss += loss.data

        for yp in y_pred.squeeze().cpu().detach().numpy():
            y_pred_list.append(yp)
        for l in batch_label_data.squeeze().cpu().detach().numpy():
            label_list.append(l)

    score = cal_roc_auc(label_list, y_pred_list)

    y_pred_list = np.array(y_pred_list)
    label_list = np.array(label_list)

    return (total_loss / n_step, score, label_list, y_pred_list)
        

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', type=str, default=None)
parser.add_argument('--save_model', type=str, default=None)
parser.add_argument('--save_result', type=str, default=None)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--model', type=str)
parser.add_argument('--n_layer_drug', type=int, default=3)
parser.add_argument('--n_layer_prot', type=int, default=3)
parser.add_argument('--n_layer_mlp', type=int, default=4)
parser.add_argument('--dim_enc_drug', type=int, default=256)
parser.add_argument('--dim_enc_prot', type=int, default=256)
parser.add_argument('--dim_mlp', type=int, default=512)

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=.1)
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--early_stopping', type=int, default=300)
parser.add_argument('--n_batch', type=int, default=48)
args = parser.parse_args()

assert args.dataset_file is not None
assert args.save_model is not None
assert args.save_result is not None

# Load data
dataset = batch.MoleculeDataset(args)
args.dim_drug_node = dataset.dim_drug_node
args.dim_drug_edge = dataset.dim_drug_edge
args.dim_prot = dataset.dim_prot_fea

train_data = dataset.load_data(partition='train')
val_data = dataset.load_data(partition='val')
test_data = dataset.load_data(partition='test')

# Setting experimental environments
nn_model = model.create_model(args)
if args.cuda:
    nn_model = nn_model.cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=10, verbose=True)

best_loss = 0
patience = 0
if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_score']

for i in range(args.n_epoch):
    t_i = time.time()
    train_loss = train(args,
                    data=train_data,
                    model=nn_model,
                    criterion=criterion,
                    optimizer=optimizer)
    val_loss, val_score, *_ = test(args, data=val_data, model=nn_model, criterion=criterion, val=True)
    scheduler.step(val_score)

    print("Time: {:.2f}, Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Score: {:.4f}, Best Score: {:.4f}".format(time.time() - t_i, i+1, train_loss, val_loss, val_score, best_loss))

    if val_score > best_loss:
        best_loss = val_score
        patience = 0
        torch.save({
            'model_state_dict': nn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_loss,
            }, args.save_model)
    else:
        patience += 1

    if patience > args.early_stopping:
        print("Early stopping")
        break

checkpoint = torch.load(args.save_model)
nn_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_loss, test_score, label_list, y_pred_list = test(args, data=test_data, model=nn_model, criterion=criterion, val=False)
print("Test Score: {:6f}".format(test_score))
np.savez(args.save_result, label=label_list, pred=y_pred_list)
