from src.differential import compute_grad
from torch import nn
import torch
import numpy as np
from typing import Tuple, Union

class BoundaryCondition:
    def __init__(
            self, 
            inputs: np.ndarray, 
            target: np.ndarray, 
            derivative_order: int = 0, 
            derivative_dim: Union[int, list[int]] = None
            ):
        """
        Args:
            inputs (np.ndarray): Boundary input points.
            target (np.ndarray): Target values for the boundary.
            derivative_order (int): Order of derivative required (0 for Dirichlet).
            derivative_dim (int): Dimension index with respect to which the derivative is taken (if needed).
        """
        self.inputs = inputs
        self.target = target
        self.derivative_order = derivative_order
        self.derivative_dim = derivative_dim

    def get_boundary_data(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert inputs to a torch tensor with gradients if needed.
        inp = torch.tensor(self.inputs, dtype=torch.float32, device=next(model.parameters()).device)
        inp = inp.clone().detach().requires_grad_(self.derivative_order > 0)
        if self.derivative_order == 0:
            # Dirichlet condition: directly compare model output.
            pred = model(inp)
        elif self.derivative_order == 1:
            # Neumann condition: compute first derivative.
            out = model(inp)
            grad = compute_grad(out, inp)
            if isinstance(self.derivative_dim, list):
                # Use index_select to extract multiple channels.
                pred = grad[:, self.derivative_dim]
            else:
                pred = grad[:, self.derivative_dim:self.derivative_dim+1]
        else:
            raise ValueError("Only derivative order 0 or 1 is supported.")
        
        target = torch.tensor(self.target, dtype=torch.float32, device=next(model.parameters()).device)

        return pred, target


class PeriodicBoundaryCondition:
    def __init__(
            self, 
            inputs_left: np.ndarray, 
            inputs_right: np.ndarray, 
            target: np.ndarray, 
            derivative_order: int = 0, 
            derivative_dim: int = None
            ):
        """
        Args:
            inputs_left (np.ndarray): Boundary points on the left side.
            inputs_right (np.ndarray): Boundary points on the right side.
            target (np.ndarray): The target difference (typically zeros).
            derivative_order (int): 0 for periodic condition on the function value,
                                    1 for periodic condition on the first derivative.
            derivative_dim (int): Dimension index for derivative if derivative_order > 0.
        """
        self.inputs_left = inputs_left
        self.inputs_right = inputs_right
        self.target = target
        self.derivative_order = derivative_order
        self.derivative_dim = derivative_dim

    def get_boundary_data(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device
        # Convert inputs to tensors and enable grad if needed.
        inp_left = torch.tensor(self.inputs_left, dtype=torch.float32, device=device).clone().detach().requires_grad_(self.derivative_order > 0)
        inp_right = torch.tensor(self.inputs_right, dtype=torch.float32, device=device).clone().detach().requires_grad_(self.derivative_order > 0)
        if self.derivative_order == 0:
            left = model(inp_left)
            right = model(inp_right)
        elif self.derivative_order == 1:
            left_out = model(inp_left)
            right_out = model(inp_right)

            output_dim = left_out.shape[1]

            grads_left = []
            grads_right = []

            for i in range(output_dim):
                grad_l = compute_grad(left_out[:, i:i+1], inp_left)[:, self.derivative_dim:self.derivative_dim+1]
                grad_r = compute_grad(right_out[:, i:i+1], inp_right)[:, self.derivative_dim:self.derivative_dim+1]

                grads_left.append(grad_l)
                grads_right.append(grad_r)

            left = torch.cat(grads_left, dim=1)
            right = torch.cat(grads_right, dim=1)
        else:
            raise ValueError("Only derivative order 0 or 1 is supported.")
    
        target = torch.tensor(self.target, dtype=torch.float32, device=device)
        pred = left-right

        return pred, target