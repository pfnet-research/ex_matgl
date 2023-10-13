from __future__ import annotations

import os
import shutil
import warnings

import torch
torch.set_default_device("cuda")

import numpy as np
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Molecule2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MGLDataLoader

from ex_matgl.graph.data import collate_fn_multi_ef
from ex_matgl.models import M3GNet_sum, M3GNet_diff
from ex_matgl.utils.training import Potential3exLossLightningModule
from ex_matgl.apps.pes import Potential3

import ase.db
from ase import Atoms
from pymatgen.core import Structure, Molecule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

def make_dataset(db_name):
    db = ase.db.connect(f"{db_name}.db")
    atoms_list = []
    structures = []
    energies = []
    forces = []
    nstate = 3
    N = db.count()
    print(f"size={N}")
    for i in range(N):
        index = i + 1
        row = db.get(id = index)
        atoms = row.symbols
        positions = row.positions
        atoms_list.append(Atoms(atoms, positions))
        mol = Molecule(species=atoms, coords=positions)
        structures.append(mol)
        energy = [(row.data["energy"]).tolist()[:nstate]]
        force = (row.data["forces"]).tolist()[:nstate]
        energies.append(energy)
        forces.append(force)
    
    labels = {"energies":energies, "forces":forces}
    print("Made dataset")
    return atoms_list, structures, labels

db_name = "heptane"
atoms_list, structures, labels = make_dataset(db_name)

element_types = get_element_list(structures)
converter = Molecule2Graph(element_types=element_types, cutoff=5.0)
    
dataset = M3GNetDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
)

nstate = 3
model = M3GNet_diff(
    element_types=element_types,
    is_intensive=False,
    ntargets=nstate,
)

atomref = np.array([-0.6054911169972964,-38.03184673889332])
lit_model =  Potential3exLossLightningModule.load_from_checkpoint("../alkane/saved_models/best_model.ckpt", model=model, element_refs=atomref)

answer = []
prediction = []
for i, test in enumerate(dataset):
    ans = test[3]
    ans = torch.squeeze(ans["energies"]).tolist()

    g = test[0].to("cuda")
    pred = lit_model(g)
    pred = pred[0].tolist()

    answer.append(ans)
    prediction.append(pred)
answer = np.array(answer)
prediction = np.array(prediction)

np.save(f"{db_name}_evaluate_answer",answer)
np.save(f"{db_name}_evaluate_prediction",prediction)
