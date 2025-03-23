from typing import Protocol, Tuple, List, Union
import numpy as np
from src.blackscholes import BlackScholesCall, BlackScholesPut
from src.boundary import BoundaryCondition, PeriodicBoundaryCondition
from src.differential import DifferentialCondition

BoundaryConditions = List[Union[BoundaryCondition, PeriodicBoundaryCondition]]

class Collocation(Protocol):
    def generate_differential_condition(self) -> DifferentialCondition:
        pass

    def generate_boundary_conditions(self) -> BoundaryConditions:
        pass


class CollocationBlackScholes:
    """
    Optimized Collocation class for Black-Scholes PDE.
    Works for both call and put options.
    """
    def __init__(self, 
                 option: Union[BlackScholesCall, BlackScholesPut], 
                 tau_max: float, 
                 S_max: float, 
                 n_collocation: int,
                 n_boundary: int):
        self.option = option
        self.tau_range = (0, tau_max)
        self.S_range = (0, S_max)
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.rng = np.random.default_rng()  

    def generate_differential_condition(self) -> DifferentialCondition:
        """
        Generate differential collocation points for the PDE.
        """
        tau = self.rng.uniform(*self.tau_range, self.n_collocation)
        S = self.rng.uniform(*self.S_range, self.n_collocation)
        X = np.column_stack((tau, S))
        y = np.zeros((self.n_collocation, 1))  # PDE residuals are zero
        return DifferentialCondition(X, y)

    def generate_boundary_conditions(self) -> BoundaryConditions:
        """
        Generate boundary condition collocation points:
        - Lower boundary: S = 0.
        - Upper boundary: S = S_max.
        - Terminal condition: tau = 0.
        
        Returns:
            A list of tuples. For each normal boundary condition, the tuple is (X_boundary, y_boundary).
        """
        # Generate random samples for tau
        tau = self.rng.uniform(*self.tau_range, self.n_boundary)

        # Condition 1 - Lower boundary: S = 0
        S_lower = np.zeros(self.n_boundary)
        lower_X = np.column_stack((tau, S_lower))
        lower_y = self.option.lower_boundary(tau=tau).reshape(-1, 1)
        bc_lower = BoundaryCondition(lower_X, lower_y)

        # Condition 2 - Upper boundary: S = S_max
        S_upper = np.full(self.n_boundary, self.S_range[1])
        upper_X = np.column_stack((tau, S_upper))
        upper_y = self.option.upper_boundary(tau=tau, S_max=self.S_range[1]).reshape(-1, 1)
        bc_upper = BoundaryCondition(upper_X, upper_y)

        # Condition 3 - Terminal condition: tau = 0
        S_terminal = self.rng.uniform(*self.S_range, self.n_boundary)
        tau_terminal = np.zeros(self.n_boundary)
        terminal_X = np.column_stack((tau_terminal, S_terminal))
        terminal_y = self.option.terminal_condition(S=S_terminal).reshape(-1, 1)
        bc_terminal = BoundaryCondition(terminal_X, terminal_y)

        return [bc_lower, bc_upper, bc_terminal]
    
class CollocationSchrodinger:
    def __init__(self, t_max: float, x_min: float, x_max: float, n_collocation: int, n_boundary: int):
        self.t_range = (0.0, t_max)
        self.x_range = (x_min, x_max)
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.rng = np.random.default_rng()

    def generate_differential_condition(self) -> DifferentialCondition:
        t = self.rng.uniform(*self.t_range, self.n_collocation)
        x = self.rng.uniform(*self.x_range, self.n_collocation)
        X = np.column_stack((t, x))
        y = np.zeros((self.n_collocation, 2))
        return DifferentialCondition(X, y)
    
    def generate_boundary_conditions(self) -> BoundaryConditions:
        """
        Generate data for the boundary conditions:
          1. Periodic boundary condition on h: enforce h(t, x_min) - h(t, x_max) = 0.
          2. Periodic boundary condition on h_x: enforce h_x(t, x_min) - h_x(t, x_max) = 0.
          3. Initial condition: h(0,x) = 2 / cosh(x).
        
        Returns:
            A list of tuples.
            For periodic conditions, the tuple is:
              ((X_left, X_right), target) where target is a numpy array of zeros with shape (n_boundary, 1).
            For the initial condition, the tuple is:
              (X_initial, y_initial)
        """

        # Shared data points for periodic conditions
        t_periodic = self.rng.uniform(*self.t_range, self.n_boundary)
        x_left = np.full(self.n_boundary, self.x_range[0])
        x_right = np.full(self.n_boundary, self.x_range[1])
        X_left = np.column_stack((t_periodic, x_left))
        X_right = np.column_stack((t_periodic, x_right))
        target_periodic = np.zeros((self.n_boundary, 2))

        # Condition 1 - Periodic condition for h
        bc_h = PeriodicBoundaryCondition(X_left, X_right, target_periodic)

        # Condition 2 - Periodic condition for h_x (spatial dervative only)
        bc_hx = PeriodicBoundaryCondition(
            X_left, X_right, target_periodic.copy(),
            derivative_order=1, derivative_dim=1
        )

        # Condition 3 - Initial condition h(0, x) = 2 / cosh(x)
        x_initial = self.rng.uniform(*self.x_range, self.n_boundary)
        t_initial = np.zeros(self.n_boundary)
        X_initial = np.column_stack((t_initial, x_initial))
        y_initial_real = 2.0 / np.cosh(x_initial)
        y_initial_imag = np.zeros_like(x_initial)
        y_initial = np.column_stack((y_initial_real, y_initial_imag))
        bc_initial = BoundaryCondition(X_initial, y_initial)

        return [
            bc_h,
            bc_hx,
            bc_initial
        ]