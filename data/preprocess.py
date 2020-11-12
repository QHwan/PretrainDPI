import numpy as np
import argparse
import prot_fea
import drug_fea

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_split', type=int, default=10)
parser.add_argument('--pretrained', type=str)
args = parser.parse_args()


if args.dataset.lower() == 'human':
    dataset_filename = './human/H_1_1.txt'
    if args.pretrained == 'transformer6':
        out_filename = './human/human_transformer6.npz'
    elif args.pretrained == 'transformer12':
        out_filename = './human/human_transformer12.npz'
    elif args.pretrained == 'transformer34':
        out_filename = './human/human_transformer34.npz'
    else:
        print("Three supported pretrained models: transformer6, transformer12, transformer34")
        exit(1)

if args.dataset.lower() == 'celegans':
    dataset_filename = './celegans/C_1_1.txt'
    if args.pretrained == 'transformer6':
        out_filename = './celegans/celegans_transformer6.npz'
    elif args.pretrained == 'transformer12':
        out_filename = './celegans/celegans_transformer12.npz'
    elif args.pretrained == 'transformer34':
        out_filename = './celegans/celegans_transformer34.npz'
    else:
        print("Three supported pretrained models: transformer6, transformer12, transformer34")
        exit(1)


# Read bare txt file
smiles_list = []
seq_list = []
label_list = []

with open(dataset_filename, 'r') as file_stream:
    for line in file_stream:
        words = line.split()
        if len(words) != 3:
            print("This line cannot be splitted with drug, protein, and label.")
            print(words)
            exit(1)
        smiles, seq, label = words
        label = int(label)

        smiles_list.append(smiles)
        seq_list.append(seq)
        label_list.append(label)

smiles_list = np.array(smiles_list)
seq_list = np.array(seq_list)
label_list = np.array(label_list)

pos_idx_list = np.where(label_list == 1)[0]
neg_idx_list = np.where(label_list == 0)[0]

assert (len(smiles_list) == len(seq_list)) and (len(seq_list) == len(label_list)) # Simple Check

# Simple data statistics
n_data = len(label_list)
n_pos = len(pos_idx_list)
n_neg = len(neg_idx_list)
prop_pos = n_pos / n_data
print("Number of pairs: {}".format(n_data))
print("Number of positive interactions: {}".format(n_pos))
print("Number of negative interactions: {}".format(n_neg))


# main process with smiles, seq, and label list.
n_data = len(label_list)

# Protein sequence: letter -> one-hot encoding 
prot_fea_list = prot_fea.get_prot_fea(seq_list, args.pretrained)

# Drug smiles: molecular graph encoding
drug_node_list, drug_edge_list, drug_n2n_list, drug_e2n_list = drug_fea.get_mol_fea(smiles_list)

# Split list
n_data_split = int(n_data / args.n_split) + 1
n_pos_data_split = int(n_data_split * prop_pos)
n_neg_data_split = n_data_split - n_pos_data_split
idx_list = list(range(n_data))
np.random.shuffle(pos_idx_list)
np.random.shuffle(neg_idx_list)
split_list = []
for i in range(args.n_split):
    buf_split_list = []
    if i == args.n_split - 1:
        buf_split_list.extend(pos_idx_list[i * n_pos_data_split : ])
        buf_split_list.extend(neg_idx_list[i * n_neg_data_split : ])
    else:
        buf_split_list.extend(pos_idx_list[i * n_pos_data_split : (i+1) * n_pos_data_split])
        buf_split_list.extend(neg_idx_list[i * n_neg_data_split : (i+1) * n_neg_data_split])

    split_list.append(buf_split_list)


np.savez(out_filename, 
    prot_fea=prot_fea_list,
    drug_node=drug_node_list, drug_edge=drug_edge_list, 
    drug_n2n=drug_n2n_list, drug_e2n=drug_e2n_list,
    label=label_list, split=split_list,
    allow_pickles=True)
