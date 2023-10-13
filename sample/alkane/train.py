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
from pytorch_lightning.callbacks import ModelCheckpoint

import matgl
from matgl.ext.pymatgen import Molecule2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MGLDataLoader

from ex_matgl.graph.data import collate_fn_multi_ef
from ex_matgl.models import M3GNet_sum, M3GNet_diff
from ex_matgl.utils.training import Potential3exLossLightningModule

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
for i in range(N):
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
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_multi_ef,
    batch_size=32,
    num_workers=0,
    generator=torch.Generator(device="cuda")
)

nstate = 3
model = M3GNet_diff(
    element_types=element_types,
    is_intensive=False,
    ntargets=nstate,
)

atomref = np.array([-0.6054911169972964,-38.03184673889332])
lit_module = Potential3exLossLightningModule(model=model, element_refs=atomref)

checkpoint_callback = ModelCheckpoint(
    dirpath="saved_models/",
    filename="best_model",
    monitor="val_Total_Loss",
    mode="min",
    save_top_k=1,
    verbose=True
)

logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(
    max_epochs=1000, 
    accelerator="gpu", 
    logger=logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
