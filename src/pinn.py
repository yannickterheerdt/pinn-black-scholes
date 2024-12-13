import torch
import torch.nn as nn
import torch.optim as optim
from src.collocation import Collocation, CollocationBlackScholes
from src.blackscholes import BlackScholesCall, BlackScholesPut, BlackScholesParams
from src.differential import Differential, BlackScholesDifferential

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = torch.relu 

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

def train_pinn(
    model: PINN,
    collocation: Collocation,
    differential: Differential,
    num_epochs: int,
    learning_rate: float,
    pde_loss_weight: float = 1.0,
    boundary_loss_weight: float = 1.0,
    device: str = "cpu"
):
    """
    Train the PINN for the Black-Scholes PDE.

    Args:
        model (PINN): The PINN model to train.
        collocation (CollocationBlackScholes): Object generating collocation points and boundary conditions.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        pde_loss_weight (float): Weight for the PDE residual loss.
        boundary_loss_weight (float): Weight for the boundary condition loss.
        device (str): Device to use ('cpu' or 'cuda').
    """
    # Move the model to the specified device
    model.to(device)

    # Loss function
    mse_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Generate collocation points for differential and boundary conditions
        differential_X, differential_y = collocation.generate_differential_data()
        boundary_conditions = collocation.generate_boundary_data()

        # Convert data to tensors and move to device
        differential_X = torch.tensor(differential_X, dtype=torch.float32, device=device)
        differential_y = torch.tensor(differential_y, dtype=torch.float32, device=device)

        boundary_tensors = [
            (torch.tensor(X, dtype=torch.float32, device=device),
             torch.tensor(y, dtype=torch.float32, device=device))
             for X, y in boundary_conditions
        ]

        # Reset gradients
        optimizer.zero_grad()

        # PDE Loss: Compute the differential loss
        differential_pred = differential.compute(model, differential_X)
        pde_loss = mse_loss(differential_pred, differential_y)

        # Boundary Loss: Compute loss for each boundary condition
        boundary_loss = 0.0
        for X, y in boundary_tensors:
            y_pred = model(X)
            boundary_loss += mse_loss(y_pred, y)

        # Weighted total loss
        total_loss = (
            pde_loss_weight * pde_loss +
            boundary_loss_weight * boundary_loss
        )

        # Backward pass and optimization step
        total_loss.backward()
        optimizer.step()

        # Print loss information every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"PDE Loss: {pde_loss.item():.6f}, "
                  f"Boundary Loss: {boundary_loss.item():.6f}, "
                  f"Total Loss: {total_loss.item():.6f}")

    print("Training completed.")

    return model


    

