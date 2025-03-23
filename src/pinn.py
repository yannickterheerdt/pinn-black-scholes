import torch
import torch.nn as nn
import torch.optim as optim
from src.collocation import Collocation
from src.differential import Differential

class PINN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_hidden_layers, 
                 activation_fn='tanh',
                 lb = None,
                 ub = None,
                 out_scale = None):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output features.
            num_hidden_layers (int): Number of hidden layers.
            activation_fn (str): Activation function to use ('tanh' or 'relu').
        """
        super(PINN, self).__init__()

        # Convert lb and ub to tensors if they are lists or tuples
        if lb is not None and not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb, dtype=torch.float32)
        if ub is not None and not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub, dtype=torch.float32)

        self.lb = lb
        self.ub = ub
        self.out_scale = out_scale

        # Input Layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        #Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Choose activation function
        if activation_fn == 'tanh':
            self.activation = torch.tanh
        elif activation_fn == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError("activation_fn must be either 'tanh' or 'relu'")

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.lb is not None and self.ub is not None:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0 

        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x) 
        
        if self.out_scale:
            x /= self.out_scale

        return x     
 
def calculate_loss(
    model: PINN,
    collocation: Collocation,
    differential: Differential,
    mse_loss: nn.Module,
    pde_loss_weight: float,
    boundary_loss_weight: float,
):
    # Generate differential condition data
    differential_condition = collocation.generate_differential_condition()
    pred_pde, target_pde = differential_condition.get_differential_data(model, differential)
    pde_loss = mse_loss(pred_pde, target_pde)

    # Generate boundary conditions data
    boundary_conditions = collocation.generate_boundary_conditions()
    boundary_loss = 0.0
    for bc in boundary_conditions:
        pred_bc, target_bc = bc.get_boundary_data(model)
        if model.out_scale:
            target_bc /= model.out_scale
        boundary_loss += mse_loss(pred_bc, target_bc)

    # Total weighted loss
    total_loss = pde_loss_weight * pde_loss + boundary_loss_weight * boundary_loss

    return total_loss, pde_loss, boundary_loss

def train_pinn(
    model: PINN,
    collocation: Collocation,
    differential: Differential,
    num_epochs: int,
    learning_rate: float,
    pde_loss_weight: float = 1.0,
    boundary_loss_weight: float = 1.0,
    optimizer_lbfgs = None,
    gamma: int = 1,
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)

    for epoch in range(num_epochs):
        # Reset gradients
        optimizer.zero_grad()

        # Calculate losses
        total_loss, pde_loss, boundary_loss = calculate_loss(
            model, collocation, differential, mse_loss,
            pde_loss_weight, boundary_loss_weight
        )

        # Backward pass and optimization step
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Print loss information every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"PDE Loss: {pde_loss.item():.6f}, "
                  f"Boundary Loss: {boundary_loss.item():.6f}, "
                  f"Total Loss: {total_loss.item():.6f}")
            
    if optimizer_lbfgs:
        print("Switching to L-BFGS optimization...")

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = calculate_loss(
                model, collocation, differential, mse_loss,
                pde_loss_weight, boundary_loss_weight
            )
            total_loss.backward()
            return total_loss

        optimizer_lbfgs.step(closure)
        print("L-BFGS optimization completed.")

    total_loss, pde_loss, boundary_loss = calculate_loss(
        model, collocation, differential, mse_loss,
        pde_loss_weight, boundary_loss_weight
    )

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"PDE Loss: {pde_loss.item():.6f}, "
          f"Boundary Loss: {boundary_loss.item():.6f}, "
          f"Total Loss: {total_loss.item():.6f}")
            

    print("Training completed.")

    return model


    

