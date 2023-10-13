"""Implementation of Interatomic Potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad

from matgl.layers import AtomRef
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import dgl
    import numpy as np

class Potential3(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 1

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | None = None,
        data_std: torch.Tensor | None = None,
        element_refs: np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_site_wise: bool = False,
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_site_wise: Enable site-wise property calculation.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_site_wise = calc_site_wise
        self.element_refs: AtomRef | None
        if element_refs is not None:
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None

        self.data_mean = torch.tensor(data_mean) if data_mean is not None else torch.zeros(1)
        self.data_std = torch.tensor(data_std) if data_std is not None else torch.ones(1)

    def forward(
        self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, l_g: dgl.DGLGraph | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        predictions = self.model(g, state_attr, l_g)
        if isinstance(predictions, tuple) and len(predictions) > 1:
            total_energies, site_wise = predictions
        else:
            total_energies = predictions
            site_wise = None
        total_energies = self.data_std * total_energies + self.data_mean
        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            property_offset = torch.unsqueeze(property_offset, -1)
            total_energies += property_offset
        
        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
        if self.calc_forces:
            nstate = total_energies.shape[-1]
            grads = torch.stack([grad(
                total_energies[...,state],
                [g.ndata["pos"], g.edata["bond_vec"]],
                grad_outputs=torch.ones_like(total_energies[...,state]),
                create_graph=True,
                retain_graph=True,
            )[0] for state in range(nstate)], axis=-1)

            forces = -grads
        
        return total_energies, forces, stresses, hessian
