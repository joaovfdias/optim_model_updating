from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import time
import warnings
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

from ..optimizer import Optimizer
from ..individual import Individual
from ..parameter import *


@dataclass
class BOConfig:
    """
    Configuration object for Bayesian Optimization.
    """

    kernel: Any = None
    alpha: float = 1e-6  # ruído/regularização
    normalize_y: bool = True
    n_restarts_optimizer: int = 5

    acquisition: str = "EI"
    xi: float = 0.01  # EI/POI
    kappa: float = 2.576  # UCB

    init_points: Optional[int] = None # computado depois conforme número de parâmetros
    batch_size: int = 1 # avaliações por iteração
    max_iter: Optional[int] = 100
    random_state: Optional[int] = None

    n_acq_candidates: int = 5000 # candidatos amostrados pelo GP para otimizar a acq_func
    n_local_perturb: int = 800 # amostras locais ao redor do melhor ponto observado

    minimize: bool = True
    normalize_X: bool = True  # *** NOVO: normaliza X no GP (como no skopt) ***

    _PRESETS = {
        "default": {
            "init_multiplier": 3,
            "min_init": 8,
            "batch_size": 1,
            "xi": 0.01,
            "n_restarts_optimizer": 5,
            "n_acq_candidates": 5000,
            "n_local_perturb": 800,
        },
        "low_budget": {
            "init_multiplier": 2,
            "min_init": 8,
            "batch_size": 1,
            "xi": 0.02,
            "n_restarts_optimizer": 3,
            "n_acq_candidates": 3000,
            "n_local_perturb": 500,
        },
        "high_cost": {
            "init_multiplier": 5,
            "min_init": 20,
            "batch_size": 1,
            "xi": 0.01,
            "n_restarts_optimizer": 8,
            "n_acq_candidates": 8000,
            "n_local_perturb": 1500,
        },
    }

    def computed_init_points(self, n_dims: int, multiplier: float = 3.0, min_points: int = 8) -> int:
        """
        Compute the number of initial points given number of dimensions.
        """
        return max(min_points, int(np.ceil(multiplier * n_dims)))

    @classmethod
    def from_preset(cls, name: str, n_dims: Optional[int] = None,
                    base: Optional["BOConfig"] = None, overwrite: bool = False) -> "BOConfig":
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls._PRESETS.keys())}")

        preset = cls._PRESETS[name].copy()
        preset_init = None
        if n_dims is not None and "init_multiplier" in preset:
            preset_init = max(preset.get("min_init", 8), int(np.ceil(preset["init_multiplier"] * n_dims)))

        default = cls()
        cfg = cls() if base is None else cls(**asdict(base))
        keys_to_apply = {k: v for k, v in preset.items() if k not in ("init_multiplier", "min_init")}

        for k, v in keys_to_apply.items():
            if not hasattr(cfg, k):
                continue
            if not overwrite and base is not None:
                if getattr(base, k) != getattr(default, k):
                    continue
            setattr(cfg, k, v)

        if preset_init is not None:
            if overwrite:
                cfg.init_points = preset_init
            else:
                if base is None or getattr(base, "init_points") == getattr(default, "init_points"):
                    cfg.init_points = preset_init

        return cfg

    def apply_preset(self, name: str, n_dims: Optional[int] = None, overwrite: bool = False) -> None:
        """
        Aplica o preset *na própria instância*, preservando campos conforme overwrite.
        """
        new_cfg = self.from_preset(name, n_dims=n_dims, base=self, overwrite=overwrite)
        for k, v in asdict(new_cfg).items():
            setattr(self, k, v)


class BO(Optimizer):
    def __init__(
        self,
        fitness_function: Callable[[List[float]], Any],
        parameters: Sequence[Parameter],
        population_size: int = 1,
        config: Optional[BOConfig] = None,
    ):
        super().__init__(fitness_function, list(parameters), population_size)

        self.config = config or BOConfig()
        self.rng = np.random.RandomState(self.config.random_state)

        self.history_X: List[np.ndarray] = []
        self.history_y: List[float] = []
        self.gp: Optional[GaussianProcessRegressor] = None

        self.parameters = list(parameters)
        self.n_dims = len(self.parameters)

        if self.config.init_points is None:
            self.config.init_points = self.config.computed_init_points(self.n_dims, multiplier=3.0, min_points=8)

        self.bounds = np.array([[getattr(p, "lower_bound"), getattr(p, "upper_bound")] for p in self.parameters], dtype=float)
        self.population: List[Individual] = []

        # scaler para normalização interna do GP
        self._fit_scaler()

    # ---------- Normalização (apenas para o GP) ----------
    def _fit_scaler(self) -> None:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        span = hi - lo
        self._x_lo = lo.astype(float)
        self._x_span = np.where(span == 0.0, 1.0, span).astype(float)

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        if not self.config.normalize_X:
            return X
        return (X - self._x_lo) / self._x_span

    def _from_unit(self, Xu: np.ndarray) -> np.ndarray:
        if not self.config.normalize_X:
            return Xu
        return self._x_lo + Xu * self._x_span

    def _vec_to_param_list(self, x: np.ndarray) -> List[float]:
        return [float(v) for v in x]

    def _param_list_to_vec(self, param_list: Sequence[float]) -> np.ndarray:
        return np.asarray(param_list, dtype=float)

    # ---------- Kernel default (em espaço normalizado) ----------
    def _default_kernel(self, y: np.ndarray):
        """
        Constant * Matern(ARD, ℓ≈0.2) + White(noise_bounds escalados por var(y))
        Trabalha naturalmente em [0,1]^d quando normalize_X=True.
        """
        n_dims = max(1, self.n_dims)
        ls0 = np.full(n_dims, 0.2, dtype=float)
        ls_bounds = [(1e-3, 1e3)] * n_dims
        var_y = float(np.var(y)) if y.size > 1 else 1.0
        nl_bounds = (max(1e-12, 1e-9 * var_y), max(1e-9, 1e2 * var_y))
        base = Matern(length_scale=ls0, length_scale_bounds=ls_bounds, nu=2.5)
        kernel = C(1.0, (1e-6, 1e6)) * base + WhiteKernel(noise_level=1e-6, noise_level_bounds=nl_bounds)
        return kernel

    # ---------- Inicialização ----------
    def initialize(self) -> List[Individual]:
        n0 = max(1, int(self.config.init_points))
        backup = getattr(self, "population_size", None)
        try:
            if backup is not None:
                self.population_size = n0
            pop = super().initial_population()
        finally:
            if backup is not None:
                self.population_size = backup

        for p in pop:
            if not isinstance(p, Individual):
                raise TypeError("Initial_population must return a list of 'Individual' objects")

        self.population = pop
        return self.population

    # ---------- Avaliação ----------
    def evaluate(self, population: Optional[List[Individual]] = None):
        pop = population or self.population
        for ind in pop:
            ind.evaluate()
            x = self._param_list_to_vec(ind.param)
            y = float(ind.fitness)
            # guarda histórico em escala ORIGINAL (deduplicate/prints/logs usam isto)
            self.history_X.append(x)
            self.history_y.append(y if self.config.minimize else -y)

    # ---------- Ajuste do GP ----------
    def _fit_gp(self):
        # treina sempre no espaço normalizado (quando normalize_X=True)
        X_raw = np.vstack(self.history_X)
        X = self._to_unit(X_raw)
        y = np.asarray(self.history_y, dtype=float)

        # warm-start do kernel otimizado anterior
        if self.gp is not None and hasattr(self.gp, "kernel_"):
            kernel0 = self.gp.kernel_
        else:
            kernel0 = self.config.kernel or self._default_kernel(y)

        self.gp = GaussianProcessRegressor(
            kernel=kernel0,
            alpha=self.config.alpha,
            normalize_y=self.config.normalize_y,
            n_restarts_optimizer=max(1, self.config.n_restarts_optimizer),
            optimizer="fmin_l_bfgs_b",
            random_state=self.rng,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.gp.fit(X, y)

    # ---------- Aquisição ----------
    def _predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.gp is not None
        Xu = self._to_unit(np.asarray(X, dtype=float))  # normaliza antes de predizer
        mu, std = self.gp.predict(Xu, return_std=True)
        return mu.reshape(-1,), std.reshape(-1,)

    def _ei(self, X: np.ndarray, best_y: float) -> np.ndarray:
        mu, sigma = self._predict_mu_sigma(X)
        sigma = np.maximum(sigma, 1e-12)
        imp = best_y - mu - self.config.xi
        Z = imp / sigma
        return imp * norm.cdf(Z) + sigma * norm.pdf(Z)

    def _ucb(self, X: np.ndarray) -> np.ndarray:
        mu, sigma = self._predict_mu_sigma(X)
        return -(mu - self.config.kappa * sigma)

    def _poi(self, X: np.ndarray, best_y: float) -> np.ndarray:
        mu, sigma = self._predict_mu_sigma(X)
        sigma = np.maximum(sigma, 1e-12)
        Z = (best_y - mu - self.config.xi) / sigma
        return norm.cdf(Z)

    def _acquisition(self, X: np.ndarray, best_y: float) -> np.ndarray:
        acq = self.config.acquisition.lower()
        if acq == 'ei':
            return self._ei(X, best_y)
        if acq == 'ucb':
            return self._ucb(X)
        if acq == 'poi':
            return self._poi(X, best_y)
        raise ValueError(f"Unknown acquisition: {self.config.acquisition}")

    # ---------- Candidatos ----------
    def _sample_candidates(self, n: int) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(n, self.n_dims))

    def _local_around_best(self, best_x: np.ndarray, n: int) -> np.ndarray:
        span = (self.bounds[:, 1] - self.bounds[:, 0])
        scale = 0.05 * span
        X = best_x + self.rng.randn(n, self.n_dims) * scale
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])

    def _deduplicate(self, X: np.ndarray, existing: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        if existing.size == 0:
            return X
        keep = []
        for i in range(X.shape[0]):
            span = self.bounds[:, 1] - self.bounds[:, 0]
            diff = (existing - X[i]) / span  # normaliza por faixa
            d = np.min(np.linalg.norm(diff, axis=1))
            if d > tol:
                keep.append(i)
        if keep:
            return X[keep]
        return X[:0]

    def _propose_batch(self, q: int) -> List[List[float]]:
        assert self.gp is not None and len(self.history_y) > 0
        X_obs = np.vstack(self.history_X)
        y_obs = np.asarray(self.history_y, dtype=float)

        best_idx = int(np.argmin(y_obs))
        best_y = float(y_obs[best_idx])
        best_x = X_obs[best_idx]

        Xc = self._sample_candidates(self.config.n_acq_candidates)
        Xc = np.vstack([Xc, self._local_around_best(best_x, self.config.n_local_perturb)])
        Xc = self._deduplicate(Xc, X_obs)
        if Xc.shape[0] == 0:
            Xc = self._sample_candidates(max(self.config.n_acq_candidates // 2, q))

        acq_vals = self._acquisition(Xc, best_y)

        span = (self.bounds[:, 1] - self.bounds[:, 0])
        span = np.where(span == 0, 1.0, span)
        hard_radius = 1e-3
        soft_sigma = 0.02

        chosen: List[int] = []
        acq_copy = acq_vals.copy()
        Xc_copy = Xc.copy()
        for _ in range(q):
            idx = int(np.argmax(acq_copy))
            chosen.append(idx)

            diff = (Xc_copy - Xc_copy[idx]) / span
            d = np.linalg.norm(diff, axis=1)

            acq_copy[d < hard_radius] = -np.inf
            acq_copy -= 0.1 * np.exp(-(d ** 2) / (2 * (soft_sigma ** 2)))

        X_new = Xc[chosen]
        return [self._vec_to_param_list(x) for x in X_new]

    # ---------- Update / Run ----------
    def update(self) -> List[Individual]:
        if len(self.history_X) == 0:
            raise RuntimeError("Call initialize() and evaluate() before update().")
        self._fit_gp()
        q = max(1, int(self.config.batch_size))
        new_param_lists = self._propose_batch(q)
        return [Individual(param=plist, fitness_function=self.fitness_function) for plist in new_param_lists]

    def run(self) -> Individual:
        self.inicio = time.time()

        pop0 = self.initialize()
        self.evaluate(pop0)
        print(
            f"\nPontos Iniciais: Melhor Fitness = {self.get_best_individual(pop0).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(pop0))}")

        for it in range(1, int(self.config.max_iter) + 1):
            new_pop = self.update()
            self.evaluate(new_pop)
            self.population.extend(new_pop)
            print(
                f"Ponto {it}: Fitness = {self.get_best_individual(new_pop).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(new_pop))}")

        best = self.get_best_individual(self.population)
        self.global_best = best

        print(f"\nMelhor solução encontrada: Fitness = {self.global_best.fitness:.4g}, Parâmetros: {self.display_parameters(self.global_best)}")
        return best


# alias com o nome pedido
BayesianOptimization = BO
