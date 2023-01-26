import re
import functools
from collections import Counter
from collections.abc import Iterable
from typing import List, Dict, Tuple

import dgl
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import torchvision as tv
from torchvision.transforms.functional import to_tensor
from PIL import Image
from rdkit import Chem
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import NPROC, default_seed, resources_dir


class MNIST_GPU(tv.datasets.MNIST):
    def __init__(self, train: bool, download: bool = True, **kwargs):
        super().__init__(
            "resources/", train=train, download=download, **kwargs)

        self.train = train
        self.imgs = [Image.fromarray(img.numpy()) for img in self.data]
        self.data = torch.cat([to_tensor(img) for img in self.imgs])

    def __getitem__(self, i):
        return i, (self.data[i], self.targets[i])

    def to(self, device):
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)
        return self


class _LazyHergCentralDataset:
    def __init__(self, split_method: str = "random"):
        self.split_method = split_method
        self._data = None
        self._split = None

    @property
    def data(self) -> Tox:
        self._init()
        return self._data

    @property
    def split(self) -> Dict[str, pd.DataFrame]:
        self._init()
        return self._split

    def _init(self):
        if self._data is not None:
            return

        label_list = retrieve_label_name_list("herg_central")
        self._data = Tox(
            name='herg_central', path=resources_dir, label_name=label_list[2])
        self._split = self._data.get_split()


hergcentral = _LazyHergCentralDataset()


class HergCentralDatasetSmi(Dataset):
    _tok_to_idx = None

    def __init__(self, data: pd.DataFrame):
        # Q) Why do we need a list of Tensors, not a 2D Tensor?
        self.smiles = [
            torch.tensor([self.tok_to_idx[tok] for tok in smi_to_tok(smiles)])
            for smiles in data["Drug"]]
        self.y = torch.tensor(data["Y"].values, dtype=torch.long)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x = self.smiles[idx]
        y = self.y[idx]
        return x, y

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        x, y = zip(*batch)
        # Shape: N (L_i, ) sized tensors -> (N, max(L_i))
        x = pad_sequence(x, batch_first=True)
        # Shape: N * (, ) -> (N, )
        y = torch.stack(y)
        return x, y

    @classmethod
    @property
    def tok_to_idx(cls) -> Dict[str, int]:
        cls._init()
        return cls._tok_to_idx

    @classmethod
    def _init(cls):
        if cls._tok_to_idx is not None:
            return

        mapping = Counter(tok for smiles in hergcentral.data.entity1
                          for tok in smi_to_tok(smiles))
        cls._tok_to_idx = {
            tok: idx for idx, (tok, _) in enumerate(mapping.most_common(), 1)}


class HergCentralDatasetGraph(Dataset):
    _atnum_to_idx = None
    _atom_classes = None

    def __init__(self, data: pd.DataFrame):
        # Again, parallel processing is a good idea.
        # Q) Why couldn't we use the batch_size arg in the previous call to
        #    Parallel?
        self.graphs: List[dgl.DGLHeteroGraph] = Parallel(
            n_jobs=NPROC, batch_size=512)(
                delayed(smi_to_graph)(smi) for smi in tqdm(data["Drug"]))
        self.y = torch.tensor(data["Y"].values, dtype=torch.long)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        x = self.graphs[idx]
        y = self.y[idx]
        return x, y

    def save(self, path: str):
        dgl.save_graphs(path, self.graphs, {"y": self.y})

    @classmethod
    def load(cls, path: str):
        graphs, labels = dgl.load_graphs(path)
        self = cls.__new__(cls)
        self.graphs = graphs
        self.y = labels["y"]
        return self

    @staticmethod
    def collate_fn(batch: List[Tuple[dgl.DGLHeteroGraph, torch.Tensor]]):
        x, y = zip(*batch)
        x = dgl.batch(x)
        # Shape: N * (, ) -> (N, )
        y = torch.stack(y)
        return x, y

    @classmethod
    @property
    def atnum_to_idx(cls) -> Dict[int, int]:
        cls._load()
        return cls._atnum_to_idx

    @classmethod
    @property
    def atom_classes(cls) -> int:
        cls._load()
        return cls._atom_classes

    @classmethod
    def _load(cls):
        if cls._atnum_to_idx is not None:
            return

        mapping = Parallel(n_jobs=NPROC)(
            delayed(get_atomic_nums)(smiles)
            for smiles in np.array_split(hergcentral.data.entity1, NPROC))
        mapping: Counter = functools.reduce(lambda x, y: x + y, mapping)

        cls._atnum_to_idx = {atnum: idx for idx, (atnum, _) in enumerate(
            mapping.most_common(), 1)}
        cls._atom_classes = len(cls._atnum_to_idx) + 1


_token_re = re.compile(r"(\[[^\]]+\]|Cl|Br|[BCNOPSFIbcnops]|.)")
_bracket_re = re.compile(r"(\d+|[A-Z][a-z]*|[a-z]+|.)")


def smi_to_tok(smiles: str):
    for tok in _token_re.finditer(smiles):
        tok = tok.group()
        if tok.startswith("["):
            for t in _bracket_re.finditer(tok):
                yield t.group()
        else:
            yield tok


def get_atomic_nums(smiles: Iterable[str]):
    return Counter(atom.GetAtomicNum()
                   for smi in smiles
                   for atom in Chem.MolFromSmiles(smi).GetAtoms())


def get_atom_feature(atom):
    features = [
        F.one_hot(
            torch.tensor(
                HergCentralDatasetGraph.atnum_to_idx[atom.GetAtomicNum()]),
            num_classes=HergCentralDatasetGraph.atom_classes),
        # Assume max 5 bonds per each atom
        F.one_hot(torch.tensor(atom.GetDegree()), num_classes=6),
        # Assume max 4 hydrogens per each atom
        F.one_hot(torch.tensor(atom.GetTotalNumHs()), num_classes=5),
        # Assume max valency of 5
        F.one_hot(torch.tensor(atom.GetImplicitValence()), num_classes=6),
        torch.tensor([atom.GetIsAromatic()], dtype=torch.float32)
    ]
    atom_feature = torch.cat(features)
    return atom_feature


_bond_type_to_idx = {
    k: i for i, k in enumerate([
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ])}


def get_bond_feature(bond):
    bt = bond.GetBondType()
    features = [
        F.one_hot(torch.tensor(_bond_type_to_idx[bt]),
                  num_classes=len(_bond_type_to_idx)),
        torch.tensor(
            [bond.GetIsConjugated(), bond.IsInRing()], dtype=torch.float32),
    ]
    bond_feature = torch.cat(features)
    return bond_feature


def smi_to_graph(smi):
    # Parse SMILES string
    mol = Chem.MolFromSmiles(smi)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    # Build graph
    bond_idxs = torch.tensor(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bonds])
    # Q) Why do we need the following line?
    bond_idxs = torch.cat([bond_idxs, bond_idxs.flip(1)])
    graph = dgl.graph(tuple(bond_idxs.T), num_nodes=len(atoms))

    # Add node features
    atom_features = torch.stack([get_atom_feature(atom) for atom in atoms])
    graph.ndata["h"] = atom_features

    # Add edge features
    bond_features = torch.stack([get_bond_feature(bond) for bond in bonds])
    bond_features = bond_features.repeat(2, 1)
    graph.edata["e_ij"] = bond_features

    return graph


def get_feature_dims():
    _dummy = Chem.MolFromSmiles("CC")
    atom_feature_dim = get_atom_feature(_dummy.GetAtomWithIdx(0)).shape[0]
    bond_feature_dim = get_bond_feature(_dummy.GetBondWithIdx(0)).shape[0]
    return atom_feature_dim, bond_feature_dim


def to_dataloader(dataset: Dataset, batch_size: int, shuffle: bool,
                  collate_fn=None, seed=default_seed):
    return DataLoader(dataset, batch_size,
                      shuffle=shuffle, collate_fn=collate_fn,
                      generator=torch.Generator().manual_seed(seed))
