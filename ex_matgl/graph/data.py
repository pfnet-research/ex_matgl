"""Tools to construct a dataset of DGL graphs."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from tqdm import trange

import matgl
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers import BondExpansion

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

def collate_fn_efs(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, line_graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor([d["energies"] for d in labels])
    f = torch.vstack([d["forces"] for d in labels])
    state_attr = torch.stack(state_attr)
    return g, l_g, state_attr, e, f, None

def collate_fn_multi_ef(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, line_graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.squeeze(torch.stack([d["energies"] for d in labels])) # change for multiple energy
    # e.shape = [batch_size, nstate]
    f = torch.vstack([torch.permute(d["forces"], (1, 2, 0)) for d in labels])
    # f.shape = [batch_size*natom, 3(xyz), nstate]
    state_attr = torch.stack(state_attr)
    return g, l_g, state_attr, e, f,None