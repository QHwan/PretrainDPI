from __future__ import print_function, division

import numpy as np
import time
import random
from collections import Counter

import torch
from torch.utils.data import Dataset

def collate_prot_data(args, data):
    prot_fea_list = data
    return prot_fea_list


def collate_drug_data(args, data):
    drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list = data

    n_tot_node = np.sum([len(x) for x in drug_node_list])
    n_tot_edge = np.sum([len(x) for x in drug_edge_list])
    for x in drug_edge_list:
        if len(x) == 0:
            n_tot_edge += 1
    dim_node = drug_node_list[0].shape[1]
    dim_edge = drug_edge_list[0].shape[1]
    n_batch = len(drug_node_list)

    idx_base_node = 0
    new_drug_node_list = np.zeros((n_tot_node, dim_node), dtype=np.float32)
    idx_drug_node_list = np.zeros(n_tot_node, dtype=np.int)
    for i, drug_node in enumerate(drug_node_list):
        n_drug_node = len(drug_node)
        new_drug_node_list[idx_base_node:idx_base_node+n_drug_node] += drug_node
        idx_drug_node_list[idx_base_node:idx_base_node+n_drug_node] += i
        idx_base_node += n_drug_node

    idx_base_edge = 0
    new_drug_edge_list = np.zeros((n_tot_edge, dim_edge), dtype=np.float32)
    idx_drug_edge_list = np.zeros(n_tot_edge, dtype=np.int)
    for i, drug_edge in enumerate(drug_edge_list):
        n_drug_edge = len(drug_edge)
        if n_drug_edge == 0:
            drug_edge = np.zeros(dim_edge, dtype=np.float32)
            n_drug_edge = 1
        new_drug_edge_list[idx_base_edge:idx_base_edge+n_drug_edge] += drug_edge
        idx_drug_edge_list[idx_base_edge:idx_base_edge+n_drug_edge] += i
        idx_base_edge += n_drug_edge

    
    idx_base = 0
    new_drug_n2n_list = np.zeros((n_tot_node, n_tot_node), dtype=np.float16)
    for i, drug_n2n in enumerate(drug_n2n_list):
        n_drug_node = len(drug_n2n)
        fancy_index = np.where(drug_n2n == 1)
        new_drug_n2n_list[idx_base : idx_base + n_drug_node,
                          idx_base : idx_base + n_drug_node] += drug_n2n
        idx_base += n_drug_node

    idx_base_node = 0
    idx_base_edge = 0
    new_drug_e2n_list = np.zeros((n_tot_node, n_tot_edge), dtype=np.float16)
    for i, drug_e2n in enumerate(drug_e2n_list):
        n_drug_node, n_drug_edge = np.shape(drug_e2n)
        if n_drug_edge == 0:
            drug_e2n = np.zeros((n_drug_node, 1), dtype=np.float16)
            n_drug_edge = 1
        new_drug_e2n_list[idx_base_node : idx_base_node + n_drug_node,
                         idx_base_edge : idx_base_edge + n_drug_edge] += drug_e2n
        idx_base_node += n_drug_node
        idx_base_edge += n_drug_edge


    return [new_drug_node_list, new_drug_edge_list,
            new_drug_n2n_list,  new_drug_e2n_list,
            idx_drug_node_list, idx_drug_edge_list]
    


def make_batch(args, data, idx):
    prot_data, drug_data, label_data = data

    prot_subdata = get_subdata(prot_data, idx)
    drug_subdata = get_subdata(drug_data, idx)
    label_subdata = get_subdata(label_data, idx)
    
    # Collate batch data
    batch_prot_data = collate_prot_data(args, data=prot_subdata)
    batch_drug_data = collate_drug_data(args, data=drug_subdata)

    batch_label_data = label_subdata

    # Cast tensor structure
    batch_prot_data = cast_tensor(batch_prot_data, dtype=['f'])
    batch_drug_data = cast_tensor(batch_drug_data, dtype=['f', 'f', 'f', 'f', 'd', 'd'])
    batch_label_data = cast_tensor(batch_label_data, dtype=['f'])

    if args.cuda:
        batch_prot_data = cast_cuda(batch_prot_data)
        batch_drug_data = cast_cuda(batch_drug_data)
        batch_label_data = cast_cuda(batch_label_data)

    return [batch_prot_data, batch_drug_data, batch_label_data]


def cast_tensor(data, dtype):
    assert isinstance(data, list)
    assert len(data) == len(dtype)
    cast_data = []
    for i, (elem, dt) in enumerate(zip(data, dtype)):
        if dt == 'f':
            cast_data.append(torch.tensor(elem).float())
        elif dt == 'd':
            cast_data.append(torch.tensor(elem).long())
        else:
            "Invalid Data Type"
            exit(1)

    return cast_data

def cast_cuda(data):
    assert isinstance(data, list)
    cast_data = [elem.cuda() for elem in data]

    if len(cast_data) == 1:
        return cast_data[0]

    return cast_data


def get_subdata(data, idx):
    assert isinstance(data, list)
    subdata = [elem[idx] for elem in data]
    return subdata


class MoleculeDataset(Dataset):
    def __init__(self, args):
        # Root dir
        #if args.dataset.lower() == 'human':
        #    dataset_filename = 'data/human/human_simple.npz'
        dataset_filename = args.dataset_file

        dataset = np.load(dataset_filename, allow_pickle=True)

        self.prot_fea_list = dataset['prot_fea']        
        self.drug_node_list = dataset['drug_node']
        self.drug_edge_list = dataset['drug_edge']
        self.drug_n2n_list = dataset['drug_n2n']
        self.drug_e2n_list = dataset['drug_e2n']
        self.label_list = dataset['label']
        self.split_list = dataset['split']

        self.data = [self.prot_fea_list,
                     self.drug_node_list, self.drug_edge_list, 
                     self.drug_n2n_list, self.drug_e2n_list,
                     self.label_list]

        self.dim_prot_fea = len(self.prot_fea_list[0]) 
        self.dim_drug_node = self.drug_node_list[0].shape[1]
        self.dim_drug_edge = self.drug_edge_list[0].shape[1]

        self.preprocess()
    

    def load_data(self, partition, add_noise=False):
        if partition == 'train':
            data = get_subdata(self.data, self.train_idx_list)
            print(Counter(data[-1]))

        elif partition == 'val':
            data = get_subdata(self.data, self.val_idx_list)
            print(Counter(data[-1]))
        
        elif partition == 'test':
            data = get_subdata(self.data, self.test_idx_list)
            print(Counter(data[-1]))

        return data


    def preprocess(self):
        n_split = len(self.split_list)
        self.train_idx_list = []
        self.val_idx_list = []
        self.test_idx_list = []

        self.test_idx_list += list(self.split_list[-1])
        self.val_idx_list += list(self.split_list[-2])
        for i in range(n_split-2):
            self.train_idx_list += list(self.split_list[i])

        assert len(set(self.train_idx_list) & set(self.val_idx_list)) == 0
        assert len(set(self.val_idx_list) & set(self.test_idx_list)) == 0
        assert len(set(self.train_idx_list) & set(self.test_idx_list)) == 0
        
