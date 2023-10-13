"""Utils for training MatGL models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from matgl.utils.training import MatglLightningModuleMixin

from ex_matgl.apps.pes import Potential3

if TYPE_CHECKING:
    import dgl
    import numpy as np
    from torch.optim import Optimizer

class Potential3LightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MatGL potentials.

    This is slightly different from the ModelLightningModel due to the need to account for energy, forces and stress
    losses.
    """

    def __init__(
        self,
        model,
        element_refs: np.ndarray | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        site_wise_weight: float = 0.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        optimizer: Optimizer | None = None,
        scheduler=None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        **kwargs,
    ):
        """
        Init PotentialLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            element_refs: element offset for PES
            energy_weight: relative importance of energy
            force_weight: relative importance of force
            stress_weight: relative importance of stress
            site_wise_weight: relative importance of additional site-wise predictions.
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            **kwargs: Passthrough to parent init.
        """
        assert energy_weight >= 0, f"energy_weight has to be >=0. Got {energy_weight}!"
        assert force_weight >= 0, f"force_weight has to be >=0. Got {force_weight}!"
        assert stress_weight >= 0, f"stress_weight has to be >=0. Got {stress_weight}!"
        assert site_wise_weight >= 0, f"site_wise_weight has to be >=0. Got {site_wise_weight}!"

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.site_wise_weight = site_wise_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha

        self.model = Potential3(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_site_wise=site_wise_weight != 0,
            data_std=self.data_std,
            data_mean=self.data_mean,
        )
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.save_hyperparameters()

    def forward(self, g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None):
        """Args:
            g: dgl Graph
            l_g: Line graph
            state_attr: State attr.

        Returns:
            energy, force, stress, h
        """
        if self.model.calc_site_wise:
            e, f, s, h, m = self.model(g=g, l_g=l_g, state_attr=state_attr)
            return e, f.float(), s, h, m

        e, f, s, h = self.model(g=g, l_g=l_g, state_attr=state_attr)
        return e, f.float(), s, h

    def step(self, batch: tuple):
        """Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        preds: tuple
        labels: tuple

        torch.set_grad_enabled(True)
        if self.model.calc_site_wise:
            g, l_g, state_attr, energies, forces, stresses, site_wise = batch
            e, f, s, _, m = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s, m)
            labels = (energies, forces, stresses, site_wise)
        else:
            g, l_g, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s)
            labels = (energies, forces, stresses)

        num_atoms = g.batch_num_nodes()
        results = self.loss_fn(
            loss=self.loss,  # type: ignore
            preds=preds,
            labels=labels,
            num_atoms=num_atoms,
        )
        batch_size = preds[0].numel()

        return results, batch_size

    def loss_fn(
        self,
        loss: nn.Module,
        labels: tuple,
        preds: tuple,
        num_atoms: int | None = None,
    ):
        """Compute losses for EFS.

        Args:
            loss: Loss function.
            labels: Labels.
            preds: Predictions
            num_atoms: Number of atoms.

        Returns::

            {
                "Total_Loss": total_loss,
                "Energy_MAE": e_mae,
                "Force_MAE": f_mae,
                "Stress_MAE": s_mae,
                "Energy_RMSE": e_rmse,
                "Force_RMSE": f_rmse,
                "Stress_RMSE": s_rmse,
            }

        """
        # labels and preds are (energy, force, stress, (optional) site_wise)    
        num_atoms = torch.tensor(num_atoms)[:,None]

        e_loss = self.loss(labels[0] / num_atoms, preds[0] / num_atoms)
        f_loss = self.loss(labels[1], preds[1])

        e_mae = self.mae(labels[0] / num_atoms, preds[0] / num_atoms)
        f_mae = self.mae(labels[1], preds[1])

        e_rmse = self.rmse(labels[0] / num_atoms, preds[0] / num_atoms)
        f_rmse = self.rmse(labels[1], preds[1])

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        m_mae = torch.zeros(1)
        m_rmse = torch.zeros(1)

        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(labels[2], preds[2])
            s_mae = self.mae(labels[2], preds[2])
            s_rmse = self.rmse(labels[2], preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_site_wise:
            m_loss = loss(labels[3], preds[3])
            m_mae = self.mae(labels[3], preds[3])
            m_rmse = self.rmse(labels[3], preds[3])
            total_loss = total_loss + self.site_wise_weight * m_loss

        return {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Site_Wise_MAE": m_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Site_Wise_RMSE": m_rmse,
        }

class Potential3exLossLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MatGL potentials.

    This is slightly different from the ModelLightningModel due to the need to account for energy, forces and stress
    losses.
    """

    def __init__(
        self,
        model,
        element_refs: np.ndarray | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        site_wise_weight: float = 0.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        optimizer: Optimizer | None = None,
        scheduler=None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        **kwargs,
    ):
        """
        Init PotentialLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            element_refs: element offset for PES
            energy_weight: relative importance of energy
            force_weight: relative importance of force
            stress_weight: relative importance of stress
            site_wise_weight: relative importance of additional site-wise predictions.
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            **kwargs: Passthrough to parent init.
        """
        assert energy_weight >= 0, f"energy_weight has to be >=0. Got {energy_weight}!"
        assert force_weight >= 0, f"force_weight has to be >=0. Got {force_weight}!"
        assert stress_weight >= 0, f"stress_weight has to be >=0. Got {stress_weight}!"
        assert site_wise_weight >= 0, f"site_wise_weight has to be >=0. Got {site_wise_weight}!"

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.site_wise_weight = site_wise_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha

        self.model = Potential3(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_site_wise=site_wise_weight != 0,
            data_std=self.data_std,
            data_mean=self.data_mean,
        )
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.save_hyperparameters()

    def forward(self, g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None):
        """Args:
            g: dgl Graph
            l_g: Line graph
            state_attr: State attr.

        Returns:
            energy, force, stress, h
        """
        if self.model.calc_site_wise:
            e, f, s, h, m = self.model(g=g, l_g=l_g, state_attr=state_attr)
            return e, f.float(), s, h, m

        e, f, s, h = self.model(g=g, l_g=l_g, state_attr=state_attr)
        return e, f.float(), s, h

    def step(self, batch: tuple):
        """Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        preds: tuple
        labels: tuple

        torch.set_grad_enabled(True)
        if self.model.calc_site_wise:
            g, l_g, state_attr, energies, forces, stresses, site_wise = batch
            e, f, s, _, m = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s, m)
            labels = (energies, forces, stresses, site_wise)
        else:
            g, l_g, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s)
            labels = (energies, forces, stresses)

        num_atoms = g.batch_num_nodes()
        results = self.loss_fn(
            loss=self.loss,  # type: ignore
            preds=preds,
            labels=labels,
            num_atoms=num_atoms,
        )
        batch_size = preds[0].numel()

        return results, batch_size

    def loss_fn(
        self,
        loss: nn.Module,
        labels: tuple,
        preds: tuple,
        num_atoms: int | None = None,
    ):
        """Compute losses for EFS.

        Args:
            loss: Loss function.
            labels: Labels.
            preds: Predictions
            num_atoms: Number of atoms.

        Returns::

            {
                "Total_Loss": total_loss,
                "Energy_MAE": e_mae,
                "Force_MAE": f_mae,
                "Stress_MAE": s_mae,
                "Energy_RMSE": e_rmse,
                "Force_RMSE": f_rmse,
                "Stress_RMSE": s_rmse,
            }

        """
        # labels and preds are (energy, force, stress, (optional) site_wise)    
        num_atoms = torch.tensor(num_atoms)[:,None]

        e_loss = self.loss(labels[0][...,0] / num_atoms, preds[0][...,0] / num_atoms)
        e_loss += self.loss(labels[0][...,1:]-labels[0][...,[0]] , preds[0][...,1:]-preds[0][...,[0]] )
        
        f_loss = self.loss(labels[1][...,0], preds[1][...,0])
        f_loss += self.loss(labels[1][...,1:]-labels[1][...,[0]] , preds[1][...,1:]-preds[1][...,[0]])

        e_mae = self.mae(labels[0] / num_atoms, preds[0] / num_atoms)
        f_mae = self.mae(labels[1], preds[1])

        e_rmse = self.rmse(labels[0] / num_atoms, preds[0] / num_atoms)
        f_rmse = self.rmse(labels[1], preds[1])

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        m_mae = torch.zeros(1)
        m_rmse = torch.zeros(1)

        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(labels[2], preds[2])
            s_mae = self.mae(labels[2], preds[2])
            s_rmse = self.rmse(labels[2], preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_site_wise:
            m_loss = loss(labels[3], preds[3])
            m_mae = self.mae(labels[3], preds[3])
            m_rmse = self.rmse(labels[3], preds[3])
            total_loss = total_loss + self.site_wise_weight * m_loss

        return {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Site_Wise_MAE": m_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Site_Wise_RMSE": m_rmse,
        }    