"""Original 37-node / 7-edge unary feature ordering."""
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

PROPERTIES = ("bp", "mp", "logP", "logPvap", "G_bonds", "H_bonds")
output_dims = {name: 1 for name in PROPERTIES}


def flatten(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from flatten(value)
        else:
            yield value


NODE_FEATURES = {'element': lambda atom, atom_idx: list(map(lambda s: int(atom.GetAtomicNum() == s), [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])), 'formal_charge': lambda atom, atom_idx: list(map(lambda s: int(atom.GetFormalCharge() == s), [-2, -1, 0, 1, 2])), 'hybridization': lambda atom, atom_idx: list(map(lambda s: int(atom.GetHybridization() == s), [1, 2, 3, 4])), 'total_num_Hs': lambda atom, atom_idx: list(map(lambda s: int(atom.GetTotalNumHs() == s), [0, 1, 2, 3])), 'aromatic_tag': lambda atom, atom_idx: atom.GetIsAromatic(), 'total_valence': lambda atom, atom_idx: list(map(lambda s: int(atom.GetTotalValence() == s), [1, 2, 3, 4, 5, 6])), 'total_degree': lambda atom, atom_idx: list(map(lambda s: int(atom.GetTotalDegree() == s), [1, 2, 3, 4])), 'is_in_ring': lambda atom, atom_idx: atom.IsInRing(), 'reaction_center': lambda atom, atom_idx: int(atom.GetIdx() == atom_idx)}

EDGE_FEATURES = {'bond_type': lambda bond, bond_idx: list(map(lambda s: int(bond.GetBondTypeAsDouble() == s), [1.0, 1.5, 2.0, 3.0])), 'is_conjugated': lambda bond, bond_idx: int(bond.GetIsConjugated()), 'rotatable': lambda bond, bond_idx: bond_smarts_query(bond, '[!$(*#*)&!D1]-!@[!$(*#*)&!D1]'), 'acylic': lambda bond, bond_idx: bond_smarts_query(bond, '*!@*'), 'reaction_center': lambda bond, bond_idx: int(bond.GetIdx() == bond_idx)}

def bond_smarts_query(bond, smarts):
    for match in bond.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        if set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) == set(match):
            return 1
    return 0

def mol_to_features(mol: Chem.rdchem.Mol, atom_idx=-1, bond_idx=-1, explicitH=False):
    if explicitH:
        mol = Chem.AddHs(mol)
    x = []
    for atom in mol.GetAtoms():
        node = []
        for key, func in NODE_FEATURES.items():
            values = func(atom, atom_idx)
            if values is list and (not np.any(values)):
                raise ValueError('No matching feature value')
            node.append(values)
        node = list(flatten(node))
        x.append(node)
    nodes = torch.tensor(np.array([np.array(xi) for xi in x]), dtype=torch.float)
    edges = []
    edge_attr = []
    for bond in mol.GetBonds():
        edges.append(np.array([[bond.GetBeginAtomIdx()], [bond.GetEndAtomIdx()]]))
        edges.append(np.array([[bond.GetEndAtomIdx()], [bond.GetBeginAtomIdx()]]))
        edge = []
        for key, func in EDGE_FEATURES.items():
            if key == 'reaction_center' and bond_idx == -1:
                continue
            values = func(bond, bond_idx)
            if values is list and (not np.any(values)):
                raise ValueError('No matching feature value')
            edge.append(values)
        edge = list(flatten(edge))
        edge_attr.extend([edge] * 2)
    if len(edges) > 0:
        edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float).reshape(-1, 7 if bond_idx == -1 else 8)
    charge = np.sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    return (nodes, edge_index, edge_attr, charge)

class UnaryMolData(Data):
    def __init__(self, smiles=None, x=None, y=None, edge_index=None,
                 edge_attr=None, charge=None):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.smiles = smiles
        self.charge = charge
        if x is not None:
            self.num_nodes = len(x)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def featurize(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecule: {smiles}")
    x, edge_index, edge_attr, charge = mol_to_features(mol)
    return UnaryMolData(smiles, x, float(target), edge_index, edge_attr, int(charge))
