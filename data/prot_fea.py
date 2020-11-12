import numpy as np
import math
import tqdm
import torch

import esm

def one_hot_encoding(x, cand_list):
    if x not in cand_list:
        print("{} is not in {}.".format(x, cand_list))
        exit(1)

    one_hot_vec = np.zeros(len(cand_list))
    one_hot_vec[cand_list.index(x)] = 1
    return(one_hot_vec)

amino_acid_list = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
amino_acid_list = list(amino_acid_list)

amino_acid_dict = {}
for i, aa in enumerate(amino_acid_list):
    amino_acid_dict[aa] = i + 1

def get_prot_fea(prot_seq_list, pretrained=None):
    if pretrained == None:
        prot_fea_list = get_prot_fea_simple(prot_seq_list, max_seq_len)
    elif pretrained.lower() == 'transformer6':
        prot_fea_list = get_prot_fea_transformer6(prot_seq_list)
    elif pretrained.lower() == 'transformer12':
        prot_fea_list = get_prot_fea_transformer12(prot_seq_list)
    elif pretrained.lower() == 'transformer34':
        prot_fea_list = get_prot_fea_transformer34(prot_seq_list)
    else:
        print("Please use good pretrained model.")
        exit(1)
    return prot_fea_list


def get_prot_fea_transformer6(prot_seq_list):
    n_prot = len(prot_seq_list)
    model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    prot_fea_list = []
    n_batch = 2
    n_step = math.ceil(n_prot / n_batch)
    for i in tqdm.tqdm(range(n_step)):
        if i == n_step:
            buf_list = prot_seq_list[i*n_batch:]
        else:
            buf_list = prot_seq_list[i*n_batch:(i+1)*n_batch]

        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j+1), buf_list[j]))
       
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq_list)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_embeddings = results['representations'][6]

        for j, (_, seq) in enumerate(batch_seq_list):
            prot_fea_list.append(token_embeddings[j, 1:len(seq)+1].mean(0).numpy())

    return prot_fea_list


def get_prot_fea_transformer12(prot_seq_list):
    n_prot = len(prot_seq_list)
    model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    prot_fea_list = []
    n_batch = 2
    n_step = math.ceil(n_prot / n_batch)
    for i in tqdm.tqdm(range(n_step)):
        if i == n_step:
            buf_list = prot_seq_list[i*n_batch:]
        else:
            buf_list = prot_seq_list[i*n_batch:(i+1)*n_batch]

        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j+1), buf_list[j]))
       
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq_list)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_embeddings = results['representations'][6]

        for j, (_, seq) in enumerate(batch_seq_list):
            prot_fea_list.append(token_embeddings[j, 1:len(seq)+1].mean(0).numpy())

    return prot_fea_list


def get_prot_fea_transformer34(prot_seq_list):
    n_prot = len(prot_seq_list)
    model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    prot_fea_list = []
    n_batch = 2
    n_step = math.ceil(n_prot / n_batch)
    for i in tqdm.tqdm(range(n_step)):
        if i == n_step:
            buf_list = prot_seq_list[i*n_batch:]
        else:
            buf_list = prot_seq_list[i*n_batch:(i+1)*n_batch]

        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j+1), buf_list[j]))
       
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq_list)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[34])
        token_embeddings = results['representations'][34]

        for j, (_, seq) in enumerate(batch_seq_list):
            prot_fea_list.append(token_embeddings[j, 1:len(seq)+1].mean(0).numpy())

    return prot_fea_list
