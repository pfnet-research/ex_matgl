from __future__ import annotations

import os
import warnings

import numpy as np
import ase.db
from ase.optimize import LBFGS, BFGS, FIRE

from ex_matgl.graph.data import collate_fn_multi_ef
from ex_matgl.models import M3GNet_diff
from ex_matgl.utils.training import Potential3exLossLightningModule
from ex_matgl.ext.ase import M3GNetCalculator3

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

element_types=["H","C"]
nstate = 3
model = M3GNet_diff(
    element_types=element_types,
    is_intensive=False,
    ntargets=nstate,
)

atomref = np.array([-0.6054911169972964,-38.03184673889332])
model_path = "../alkane/saved_models/best_model.ckpt"
lit_model =  Potential3exLossLightningModule.load_from_checkpoint(model_path, model=model, element_refs=atomref)
pot = lit_model.model.to('cpu')

state = 1
Calculator = M3GNetCalculator3(potential=pot,state=state)

db = ase.db.connect(f"heptane.db")
row = db.get(id=1)
atoms = row.toatoms()

os.makedirs(f"optimize_structures", exist_ok=True)
atoms.calc = Calculator
opt = BFGS(atoms, trajectory=f"optimize_structures/bfgs_heptane.traj")
opt.run(fmax=0.01)

