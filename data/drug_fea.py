import numpy as np
import tqdm
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import networkx as nx


def one_hot_encoding(x, cand_list):
    if x not in cand_list:
        print("{} is not in {}.".format(x, cand_list))
        exit(1)

    one_hot_vec = np.zeros(len(cand_list))
    one_hot_vec[cand_list.index(x)] = 1
    return list(one_hot_vec)

atom_type_list = ['C', 'N', 'O', 'S', 'Br', 'Cl', 'N', 'P', 'F', 'I', 'B', 'Si', 'H', 'Na', 'Ca', 'Se', 'Fe', 'Al',
                  'Pt', 'Bi', 'Au', 'Hg', 'Gd', 'Ge', 'K', 'Co', 'Sr', 'In', 'Mn', 'Ag', 'Mg',
                  'Cu', 'Zn', 'As', 'Ni', 'V', 'Zr', 'Sn', 'Li', 'Sb', 'Pd', 'Ti', 'Ho', 'Ru',
                  'Rh', 'Cr', 'Ga', 'Tb', 'Ir', 'Te', 'Pb', 'W', 'Cs', 'Mo', 'Re', 'U', 'Tl', 'Ac', 'Ba', 'Cf', 'Cd', 'Ar',
                  'Rb', 'Ce', 'Ta', 'Be', 'Po', 'Y', 'Fr', 'Tc', 'He']
valence_list = [0, 1, 2, 3, 4, 5, 6]
formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
radical_list = [0, 1, 2]
hybridization_list = [Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.UNSPECIFIED,]
aromatic_list = [0, 1]
num_h_list = [0, 1, 2, 3, 4]
degree_list = [0, 1, 2, 3, 4, 5]

bond_type_list = [Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC]
conjugate_list = [0, 1]
ring_list = [0, 1]
stereo_list = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOCIS,
               Chem.rdchem.BondStereo.STEREOTRANS,
               Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ
               ]

def get_atom_feature(atom, explicit_H=False):
    out = one_hot_encoding(atom.GetSymbol(),
                            atom_type_list)
    out += one_hot_encoding(atom.GetHybridization(),
                            hybridization_list)
    if not explicit_H:
        out += one_hot_encoding(atom.GetTotalNumHs(),
                                num_h_list)

    out += one_hot_encoding(atom.GetFormalCharge(), formal_charge_list)
    out += [int(atom.GetIsAromatic())]

    return(out)

def bond_features(bond, explicit_H=False):
    out = one_hot_encoding(bond.GetBondType(),
                            bond_type_list)
    out += [bond.GetIsConjugated()]
    out += [bond.IsInRing()]
    return(out)


def get_mol_fea(drug_smiles_list):
    drug_node_list = []
    drug_edge_list = []
    drug_n2n_list = []
    drug_e2n_list = []
    node_dim = len(atom_type_list + hybridization_list + num_h_list+formal_charge_list) + 1
    edge_dim = len(bond_type_list) + 1 + 1

    for i, smiles in tqdm.tqdm(enumerate(drug_smiles_list), total=len(drug_smiles_list)):
        if smiles[-1] == ' ':
            smiles = smiles[:-1]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("molecule {} is not defined well".format(smiles))
            exit(1)

        atom_list = mol.GetAtoms()
        bond_list = mol.GetBonds()

        n_node = len(atom_list)
        n_edge = len(bond_list)

        node = np.zeros((n_node, node_dim))
        for j, atom in enumerate(atom_list):
            node[j] += get_atom_feature(atom)
        node = np.array(node)

        n2n = GetAdjacencyMatrix(mol)

        edge = np.zeros((n_edge, edge_dim))
        e2n = np.zeros((n_node, n_edge))
        edge_idx = 0
        for j in range(n_node):
            for k in range(j+1, n_node):
                bond = mol.GetBondBetweenAtoms(j, k)
                if bond is not None:
                    edge[edge_idx] += bond_features(bond)
                    e2n[j, edge_idx] += 1
                    e2n[k, edge_idx] += 1
                    edge_idx += 1

        drug_node_list.append(node)
        drug_n2n_list.append(n2n)
        drug_edge_list.append(edge)
        drug_e2n_list.append(e2n)

    return (drug_node_list, drug_edge_list, 
            drug_n2n_list, drug_e2n_list)
