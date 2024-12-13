import torch
from torch import nn
from src.blackscholes import BlackScholesCall, BlackScholesPut
from typing import Protocol, Union

class Differential(Protocol):
    def compute(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the residuals of the PDE for the given model and inputs.

        Args:
            model (nn.Module): The PINN model approximating the solution.
            inputs (torch.Tensor): Tensor of shape (batch_size, input_dim), containing the collocation points.

        Returns:
            torch.Tensor: Tensor of residuals for the PDE.
        """
        ...

class BlackScholesDifferential(Differential):
    def __init__(self, option: Union[BlackScholesCall, BlackScholesPut]):
        self.option = option

    def compute(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.requires_grad:
            inputs = inputs.requires_grad_()
        V = model(inputs)
        tau, S = inputs[:, 0:1], inputs[:, 1:2]
        r, sigma = self.option.params.r, self.option.params.sigma

        dV_dX = torch.autograd.grad(outputs=V, inputs=inputs, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        dV_dtau, dV_dS = dV_dX[:,0:1], dV_dX[:, 1:2]
        dV_dS2 = torch.autograd.grad(outputs=dV_dS, inputs=inputs, grad_outputs=torch.ones_like(dV_dS), retain_graph=True)[0][:, 1:2]

        return -dV_dtau + 0.5 * sigma**2 * S**2 * dV_dS2 + r * S * dV_dS - r * V



