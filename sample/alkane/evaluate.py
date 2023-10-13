from __future__ import annotations

import os
import shutil
import warnings
from tqdm import tqdm

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

db = ase.db.connect("alkane.db")
atoms_list = []
structures = []
energies = []
forces = []

N = db.count()
for i in tqdm(range(N)):
    index = i + 1
    row = db.get(id = index)
    atoms = row.symbols
    positions = row.positions
    atoms_list.append(Atoms(atoms, positions))
    mol = Molecule(species=atoms, coords=positions)
    structures.append(mol)
    energy = [(row.data["energy"]).tolist()]
    force = (row.data["forces"]).tolist()
    energies.append(energy)
    forces.append(force)
print("Made dataset")
labels = {"energies":energies, "forces":forces}

element_types = get_element_list(structures)
converter = Molecule2Graph(element_types=element_types, cutoff=5.0)

dataset = M3GNetDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
)
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)

nstate = 3
model = M3GNet_diff(
    element_types=element_types,
    is_intensive=False,
    ntargets=nstate,
)
atomref = np.array([-0.6054911169972964,-38.03184673889332])
lit_model =  Potential3exLossLightningModule.load_from_checkpoint("saved_models/best_model.ckpt", model=model, element_refs=atomref)

answer = []
prediction = []
for data in [train_data, val_data, test_data]:
    tmp_ans = []
    tmp_pred = []
    for test in tqdm(data):
        ans = test[3]
        ans = torch.squeeze(ans["energies"]).tolist()

        g = test[0].to("cuda")
        pred = lit_model(g)
        pred = pred[0].tolist()

        tmp_ans.append(ans)
        tmp_pred.append(pred)
    answer.append(np.array(tmp_ans))
    prediction.append(np.array(tmp_pred))
    
np.savez("evaluate_answer",train = answer[0], val=answer[1], test=answer[2])
np.savez("evaluate_prediction",train = prediction[0], val=prediction[1], test=prediction[2])
