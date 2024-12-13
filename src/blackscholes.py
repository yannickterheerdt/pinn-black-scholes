from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from typing import Union, Tuple

@dataclass
class BlackScholesParams:
    """
    Parameters for the Black-Scholes model.
    """
    K: float     # Strike price
    r: float     # Risk-free interest rate (annualized)
    sigma: float # Volatility of the underlying stock (annualized)

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")

    def calculate_d1_d2(self, S: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate d1 and d2 terms for the Black-Scholes formula.

        :param S: Stock price(s), must be positive.
        :param tau: Time to maturity (in years), must be non-negative.
        :return: Tuple of d1 and d2 values as NumPy arrays.
        """
        d1 = np.where(
            S > 0,
            (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau)),
            -np.inf
        )
        d2 = d1 - self.sigma * np.sqrt(tau)
        return d1, d2


class BlackScholesCall:
    """
    Black-Scholes model for a call option.
    """
    def __init__(self, params: BlackScholesParams):
        self.params = params

    def payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Payoff at maturity for a call option.
        """
        return np.maximum(S - self.params.K, 0)

    def price(self, S: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Price of the call option.
        """
        d1, d2 = self.params.calculate_d1_d2(S, tau)
        return S * norm.cdf(d1) - self.params.K * np.exp(-self.params.r * tau) * norm.cdf(d2)

    def lower_boundary(self, tau: np.ndarray) -> np.ndarray:
        """
        Lower boundary condition: V(0, t) = 0.
        """
        return np.zeros_like(tau)

    def upper_boundary(self, tau: np.ndarray, S_max: float) -> np.ndarray:
        """
        Upper boundary condition: V(S, t) -> S - K e^{-r \tau}.
        """
        return S_max - self.params.K * np.exp(-self.params.r * tau)

    def terminal_condition(self, S: np.ndarray) -> np.ndarray:
        """
        Terminal condition: V(S, 0) = max(S - K, 0).
        """
        return self.payoff(S)


class BlackScholesPut:
    """
    Black-Scholes model for a put option.
    """
    def __init__(self, params: BlackScholesParams):
        self.params = params

    def payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Payoff at maturity for a put option.
        """
        return np.maximum(self.params.K - S, 0)

    def price(self, S: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Price of the put option.
        """
        d1, d2 = self.params.calculate_d1_d2(S, tau)
        return self.params.K * np.exp(-self.params.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def lower_boundary(self, tau: np.ndarray) -> np.ndarray:
        """
        Lower boundary condition: V(0, t) = K e^{-r \tau}.
        """
        return self.params.K * np.exp(-self.params.r * tau)

    def upper_boundary(self, tau: np.ndarray, S_max: float) -> np.ndarray:
        """
        Upper boundary condition: V(S, t) -> 0.
        """
        return np.zeros_like(tau)

    def terminal_condition(self, S: np.ndarray) -> np.ndarray:
        """
        Terminal condition: V(S, 0) = max(K - S, 0).
        """
        return self.payoff(S)


def calculate_option_price(option: Union[BlackScholesCall, BlackScholesPut], 
                           S: Union[float, np.ndarray], 
                           tau: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the option price for given stock price(s) and time to maturity.

    :param option: An instance of BlackScholesCall or BlackScholesPut.
    :param S: Stock price(s), can be a float or NumPy array.
    :param tau: Time to maturity (in years), can be a float or NumPy array.
    :return: Option price(s) as a NumPy array.
    """
    S = np.asarray(S)
    tau = np.asarray(tau)

    payoff = option.payoff(S)
    price = option.price(S, tau)

    return np.where(tau == 0, payoff, price)