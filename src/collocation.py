from typing import Protocol, Tuple, List, Union
import numpy as np
from src.blackscholes import BlackScholesCall, BlackScholesPut


class Collocation(Protocol):
    def generate_differential_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def generate_boundary_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass


class CollocationBlackScholes:
    """
    Generalized Collocation class for Black-Scholes PDE.
    Works for both call and put options.
    """
    def __init__(self, 
                 option: Union[BlackScholesCall, BlackScholesPut], 
                 t_max: float, 
                 S_max: float, 
                 n: int):
        self.option = option
        self.tau_range = (0, t_max)
        self.S_range = (0, S_max)
        self.n = n

    def generate_differential_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate differential collocation points for the PDE.
        """
        X = np.concatenate([
            np.random.uniform(*self.tau_range, (self.n, 1)),  # Random times
            np.random.uniform(*self.S_range, (self.n, 1))  # Random stock prices
        ], axis=1)
        y = np.zeros((self.n, 1))  # PDE residuals are zero
        return X, y

    def generate_boundary_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate boundary condition collocation points, including:
        - Lower boundary: S = 0.
        - Upper boundary: S = S_max.
        - Terminal condition: tau = 0.
        """
        tau = np.random.uniform(*self.tau_range, (self.n, 1))
        S_lower = np.zeros((self.n, 1))
        lower_boundary_X = np.concatenate([tau, S_lower], axis=1)
        lower_boundary_y = self.option.lower_boundary(tau=tau).reshape(-1, 1)

        S_upper = np.full((self.n, 1), self.S_range[1])
        upper_boundary_X = np.concatenate([tau, S_upper], axis=1)
        upper_boundary_y = self.option.upper_boundary(tau=tau, S_max=self.S_range[1]).reshape(-1, 1)

        S_terminal = np.random.uniform(*self.S_range, (self.n, 1))
        tau_terminal = np.zeros((self.n, 1))
        terminal_condition_X = np.concatenate([tau_terminal, S_terminal], axis=1)
        terminal_condition_y = self.option.terminal_condition(S=S_terminal).reshape(-1, 1)

        return [
            (lower_boundary_X, lower_boundary_y),
            (upper_boundary_X, upper_boundary_y),
            (terminal_condition_X, terminal_condition_y)
        ]