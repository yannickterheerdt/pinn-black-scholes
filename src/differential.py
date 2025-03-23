import torch
from torch import nn
import numpy as np
from src.blackscholes import BlackScholesCall, BlackScholesPut
from typing import Protocol, Union, Tuple

def compute_grad(
    output: torch.Tensor,
    inputs: torch.Tensor,
    retain_graph: bool = True,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute the gradient of `output` with respect to `inputs`.

    Args:
        output (torch.Tensor): Tensor whose gradient is computed.
        inputs (torch.Tensor): Tensor with respect to which the gradient is computed.
        retain_graph (bool): Whether to retain the computation graph.
        create_graph (bool): Whether to construct the graph of the derivative.

    Returns:
        torch.Tensor: The computed gradient.
    """
    return torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph
    )[0]

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
        V = model(inputs)
        tau, S = inputs[:, 0:1], inputs[:, 1:2]
        r, sigma = self.option.params.r, self.option.params.sigma

        dV_dX = compute_grad(V, inputs)
        dV_dtau, dV_dS = dV_dX[:,0:1], dV_dX[:, 1:2]
        dV_dS2 =  dV_dS2 = compute_grad(dV_dS, inputs, create_graph=False)[:, 1:2]

        return -dV_dtau + 0.5 * sigma**2 * S**2 * dV_dS2 + r * S * dV_dS - r * V
    
class SchrodingerDifferential(Differential):
    def compute(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the residual for the Schrödinger equation
            i * h_t + 0.5 * h_xx + |h|^2 * h = 0,
        where h = u + i*v, with u and v being the real and imaginary parts of the solution.
        
        Args:
            model (nn.Module): The PINN model returning a tensor of shape (batch, 2) for u and v.
            inputs (torch.Tensor): Tensor of shape (batch, 2) where the columns represent t and x, respectively.
        
        Returns:
            torch.Tensor: Tensor of shape (batch, 2) containing the residuals' real and imaginary parts.
        """

        # Model returns h = [u, v]
        h = model(inputs)
        u, v = h[:, 0:1], h[:, 1:2]

        # Compute full gradients for u and v.
        du = compute_grad(u, inputs)
        u_t, u_x = du[:, 0:1], du[:, 1:2]
    
        dv = compute_grad(v, inputs)
        v_t, v_x = dv[:, 0:1], dv[:, 1:2]

        # Compute second spatial derivatives 
        u_xx = compute_grad(u_x, inputs)[:, 1:2]
        v_xx = compute_grad(v_x, inputs)[:, 1:2]

        # Compute |h|^2 = u^2 + v^2.
        abs_h_sq = u**2 + v**2

        # Construct the PDE residual:
        f_real = -v_t + 0.5 * u_xx + abs_h_sq * u
        f_imag = u_t + 0.5 * v_xx + abs_h_sq * v

        return torch.cat([f_real, f_imag], dim=1)
    
class DifferentialCondition:
    def __init__(self, inputs: np.ndarray, target: np.ndarray):
        """
        Args:
            inputs (np.ndarray): Collocation points for the differential operator.
            target (np.ndarray): Target values for the residual (typically zeros).
        """
        self.inputs = inputs
        self.target = target

    def get_differential_data(self, model: nn.Module, differential: Differential) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert inputs and target to tensors on the correct device.
        device = next(model.parameters()).device
        inp = torch.tensor(self.inputs, dtype=torch.float32, device=device)
        target = torch.tensor(self.target, dtype=torch.float32, device=device)
        # Ensure that gradients are enabled for differential computation.
        if not inp.requires_grad:
            inp = inp.requires_grad_()
        pred = differential.compute(model, inp)
        return pred, target
