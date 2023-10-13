"""Interfaces to the Atomic Simulation Environment package for dynamic simulations."""

from __future__ import annotations

import collections
import contextlib
import io
import pickle
import sys
from enum import Enum
from typing import TYPE_CHECKING, Literal

import ase.optimize as opt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.optimization.neighbors import find_points_in_spheres

import matgl
from matgl.graph.converters import GraphConverter
from matgl.ext.ase import Atoms2Graph

if TYPE_CHECKING:
    import dgl
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer

    from matgl.apps.pes import Potential

class M3GNetCalculator3(Calculator):
    """M3GNet calculator for ASE."""

    implemented_properties = ("energy", "free_energy", "forces", "stress", "hessian")

    def __init__(
        self,
        potential: Potential,
        state: int = 0,
        state_attr: torch.Tensor | None = None,
        stress_weight: float = 1.0,
        **kwargs,
    ):
        """
        Init M3GNetCalculator with a Potential.

        Args:
            potential (Potential): m3gnet.models.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.state = state
        self.compute_stress = potential.calc_stresses
        self.compute_hessian = potential.calc_hessian
        self.stress_weight = stress_weight
        self.state_attr = state_attr
        self.element_types = potential.model.element_types  # type: ignore
        self.cutoff = potential.model.cutoff

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        graph, state_attr_default = Atoms2Graph(self.element_types, self.cutoff).get_graph(atoms)  # type: ignore
        if self.state_attr is not None:
            energies, forces, stresses, hessians = self.potential(graph, self.state_attr)
        else:
            energies, forces, stresses, hessians = self.potential(graph, state_attr_default)
        energies = energies[self.state]
        forces = forces[...,self.state]
        self.results.update(
            energy=energies.detach().cpu().numpy(),
            free_energy=energies.detach().cpu().numpy(),
            forces=forces.detach().cpu().numpy(),
        )
        if self.compute_stress:
            self.results.update(stress=stresses.detach().cpu().numpy() * self.stress_weight)
        if self.compute_hessian:
            self.results.update(hessian=hessians.detach().cpu().numpy())