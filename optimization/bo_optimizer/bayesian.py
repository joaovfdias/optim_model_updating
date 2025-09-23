from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import time
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b

# Import your package classes
from ..optimizer import Optimizer
from ..individual import Individual
from ..parameter import *


# ==================
# CONFIGURAÇÃO DO BO
# ==================
@dataclass
class BOConfig:
    """
    Configuration object for Bayesian Optimization.
    """

    # GP / surrogate
    kernel: Any = None
    alpha: float = 1e-6              # regularização numérica (ruído)
    normalize_y: bool = True
    n_restarts_optimizer: int = 5    # reinícios para otimizar hiperparâmetros do GP
    normalize_X: bool = True         # normaliza X para [0,1]^d no GP

    # aquisição
    acquisition: str = "EI"
    xi: float = 0.01                 # EI/POI
    kappa: float = 2.576             # UCB

    # otimização da aquisição
    acq_optimizer: str = "lbfgs"  # "sampling" (tradicional) | "lbfgs"
    # lbfgs:
    acq_n_points: int = 10000        # pontos para pré-seleção
    acq_n_restarts: int = 5          # multi-starts
    acq_maxiter: int = 20            # iterações
    # sampling:
    n_acq_candidates: int = 5000
    n_local_perturb: int = 800

    # loop
    init_points: Optional[int] = None
    init_multiplier = 5
    init_min = 10
    batch_size: int = 1
    # max_iter: Optional[int] = 100 | passar para run
    random_state: Optional[int] = None

    # objetivo
    minimize: bool = True

    # presets de custo
    _PRESETS = {
        "low": {
            "init_multiplier": 3,
            "min_init": 8,
            "batch_size": 1,
            "xi": 0.02,
            "n_restarts_optimizer": 3,
            "n_acq_candidates": 3000,
            "n_local_perturb": 500,
            "acq_n_points": 6000,
            "acq_n_restarts": 3,
            "acq_maxiter": 1000,
        },
        "high": {
            "init_multiplier": 5,
            "min_init": 20,
            "batch_size": 1,
            "xi": 0.01,
            "n_restarts_optimizer": 8,
            "n_acq_candidates": 8000,
            "n_local_perturb": 1500,
            "acq_n_points": 20000,
            "acq_n_restarts": 10,
            "acq_maxiter": 40,
        },
    }

    def computed_init_points(self, n_dims: int, multiplier: float = 3.0, min_points: int = 8) -> int:
        return max(min_points, int(np.ceil(multiplier * n_dims)))

    @classmethod
    def from_preset(cls, name: str, n_dims: Optional[int] = None, base: Optional["BOConfig"] = None, overwrite: bool = False) -> "BOConfig":
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls._PRESETS.keys())}")

        preset = cls._PRESETS[name].copy()
        preset_init = None
        if n_dims is not None and "init_multiplier" in preset:
            preset_init = max(preset.get("min_init", 8), int(np.ceil(preset["init_multiplier"] * n_dims)))

        default = cls()
        cfg = cls() if base is None else cls(**asdict(base))
        # keys_to_apply = {k: v for k, v in preset.items() if k not in ("init_multiplier", "min_init")}
        keys_to_apply = {k: v for k, v in preset.items()}


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
        new_cfg = self.from_preset(name, n_dims=n_dims, base=self, overwrite=overwrite)
        for k, v in asdict(new_cfg).items():
            setattr(self, k, v)


# =====================
# BAYESIAN OPTIMIZATION
# =====================
class BO(Optimizer):
    def __init__(
        self,
        fitness_function,
        parameters: Sequence[Parameter],
        initial_points: int = None,
        config: Optional[BOConfig] = None,
    ):

        self.config = config or BOConfig()
        self.n_dims = len(list(parameters))

        if not initial_points and not self.config.init_points:
            initial_points = self.config.computed_init_points(self.n_dims)
        else:
            initial_points = initial_points or self.config.init_points

        super().__init__(fitness_function, list(parameters), population_size=initial_points)

        self.rng = np.random.RandomState(self.config.random_state)

        self.history_X: List[np.ndarray] = []
        self.history_y: List[float] = []
        self.gp: Optional[GaussianProcessRegressor] = None

        self.parameters = list(parameters)

        self.bounds = np.array([[getattr(p, "lower_bound"), getattr(p, "upper_bound")] for p in self.parameters], dtype=float)
        self.population: List[Individual] = []
        self.best = None

        # scaler (normalização interna do GP)
        self._fit_scaler()

    # -----------------
    # Normalização (GP)
    # -----------------
    def _fit_scaler(self) -> None:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        span = hi - lo
        self._x_lo = lo.astype(float)
        self._x_span = np.where(span == 0.0, 1.0, span).astype(float)

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        # [0,1]^d quando normalize_X=True
        if not self.config.normalize_X:
            return X
        Xu = (X - self._x_lo) / self._x_span
        return np.clip(Xu, 0.0, 1.0)

    def _from_unit(self, Xu: np.ndarray) -> np.ndarray:
        Xu = np.clip(Xu, 0.0, 1.0)
        if not self.config.normalize_X:
            return Xu
        return self._x_lo + Xu * self._x_span  # Xu * span + lo

    def _vec_to_param_list(self, x: np.ndarray) -> List[float]:
        return [float(v) for v in x]

    def _param_list_to_vec(self, param_list: Sequence[float]) -> np.ndarray:
        return np.asarray(param_list, dtype=float)

    # Kernel default (em espaço normalizado)
    def _default_kernel(self, y: np.ndarray):
        """
        Constant * Matern(ARD, ℓ≈0.2) + White(noise_bounds escalados por var(y)).
        """
        n_dims = max(1, self.n_dims)
        ls0 = np.full(n_dims, 0.2, dtype=float)
        ls_bounds = [(1e-2, 1e2)] * n_dims # testar 1e-3, 1e3
        var_y = float(np.var(y)) if y.size > 1 else 1.0
        nl_bounds = (max(1e-12, 1e-9 * var_y), max(1e-9, 1e2 * var_y))
        base = Matern(length_scale=ls0, length_scale_bounds=ls_bounds, nu=2.5)
        kernel = C(1.0, (1e-6, 1e6)) * base + WhiteKernel(noise_level=1e-6, noise_level_bounds=nl_bounds)
        return kernel

    # -------------
    # Inicialização
    # -------------
    def initialize(self) -> List[Individual]:

        pop = super().initial_population()

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
            ind.data = ind.data or {}

            x = self._param_list_to_vec(ind.param)
            y = float(ind.fitness)

            # guarda histórico em escala ORIGINAL (deduplicate/prints/logs usam isto)
            self.history_X.append(x)
            self.history_y.append(y if self.config.minimize else -y)

            # previsão do GP no ponto avaliado (se já houver GP ajustado)
            if self.gp is not None:
                mu, sigma = self._predict_mu_sigma(x.reshape(1, -1))
                ind.data["pred_mu"] = float(mu[0])
                ind.data["pred_sigma"] = float(sigma[0])

    # ------------
    # Ajuste do GP
    # ------------
    def _fit_gp(self):
        # treina sempre no espaço normalizado (quando normalize_X=True)
        X_raw = np.vstack(self.history_X)
        X = self._to_unit(X_raw)
        y = np.asarray(self.history_y, dtype=float)

        # filtro de linhas válidas
        # ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        # X, y = X[ok], y[ok]

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

    # --------------
    # Predição do GP
    # --------------
    def _predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.gp is not None
        Xu = self._to_unit(np.asarray(X, dtype=float))  # normaliza antes de predizer
        if not np.isfinite(Xu).all():
            raise ValueError("Predict recebeu X com NaN/inf.")
        mu, std = self.gp.predict(Xu, return_std=True)
        std = np.maximum(std, 1e-12)  # evita divisão por zero no EI/PI
        return mu.reshape(-1,), std.reshape(-1,)

    # --------------------------------
    # Diagnóstico do kernel (para log)
    # --------------------------------
    def _extract_length_scales(self, kernel_obj) -> Optional[List[float]]:
        if hasattr(kernel_obj, "length_scale"):
            ls = np.atleast_1d(kernel_obj.length_scale).astype(float)
            return ls.tolist()
        # varre k1/k2 recursivamente até achar length_scale
        for attr in ("k1", "k2"):
            if hasattr(kernel_obj, attr):
                ls = self._extract_length_scales(getattr(kernel_obj, attr))
                if ls is not None:
                    return ls
        return None

    def _kernel_diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.gp is not None and hasattr(self.gp, "kernel_"):
            k = self.gp.kernel_
            d["kernel_str"] = str(k)
            d["length_scales"] = self._extract_length_scales(k)
            try:
                d["signal_variance"] = float(k.k1.k1.constant_value)
            except Exception:
                pass
            try:
                d["noise_level"] = float(k.k2.noise_level)
            except Exception:
                pass
            d["log_marginal_likelihood"] = float(getattr(self.gp, "log_marginal_likelihood_value_", np.nan))
        return d

    # --------------------
    # Funções de Aquisição
    # --------------------
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
        acq = self.config.acquisition.upper()
        if acq == 'EI':
            return self._ei(X, best_y)
        if acq == 'UCB':
            return self._ucb(X)
        if acq == 'POI':
            return self._poi(X, best_y)
        raise ValueError(f"Unknown acquisition: {self.config.acquisition}")

    # ---------------------
    # Candidatos (sampling)
    # ---------------------
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

    # ----------------------------------
    # Otimização da aquisição por L-BFGS
    # ----------------------------------
    def _optimize_acq_lbfgs(self, best_y: float, q: int) -> np.ndarray:
        """
        Otimiza a aquisição em [0,1]^d com L-BFGS-B (multi-start) e retorna q pontos em escala ORIGINAL.
        Estratégia: pré-amostrar acq em acq_n_points, pegar top-k como seeds e refinar via L-BFGS.
        """
        # 1) pré-amostra seeds em [0,1]^d
        npts = int(max(self.config.acq_n_points, q * 100))
        Xu = self.rng.rand(npts, self.n_dims)                  # seeds no espaço normalizado
        X0 = self._from_unit(Xu)                               # escala original para avaliar aquisição
        vals = self._acquisition(X0, best_y)                   # maior é melhor

        # 2) selecione multi-starts
        k = int(max(self.config.acq_n_restarts, q))
        starts = Xu[np.argsort(-vals)[:k]]

        bounds_unit = [(0.0, 1.0)] * self.n_dims

        def obj(z_unit: np.ndarray) -> Tuple[float, np.ndarray]:
            X_real = self._from_unit(z_unit.reshape(1, -1))
            val = self._acquisition(X_real, best_y)[0]
            return -float(val), None  # L-BFGS minimiza; grad numérico implícito

        chosen = []
        tried = []

        # 3) roda L-BFGS a partir de cada seed
        for x0 in starts:
            xopt, f, _ = fmin_l_bfgs_b(func=obj, x0=x0, bounds=bounds_unit, maxiter=self.config.acq_maxiter)
            tried.append((xopt.copy(), -f))

        # 4) ordena por valor de aquisição e aplica diversidade simples
        tried.sort(key=lambda t: t[1], reverse=True)
        hard_radius = 1e-3  # em [0,1]^d
        for xopt, _score in tried:
            if len(chosen) >= q:
                break
            if not chosen:
                chosen.append(xopt)
            else:
                d = np.min([np.linalg.norm(xopt - c) for c in chosen])
                if d > hard_radius:
                    chosen.append(xopt)

        Xu_best = np.array(chosen[:q])
        return self._from_unit(Xu_best)

    # ----------------
    # Seleção do batch
    # ----------------
    def _propose_batch(self, q: int) -> List[List[float]]:
        assert self.gp is not None and len(self.history_y) > 0
        X_obs = np.vstack(self.history_X)
        y_obs = np.asarray(self.history_y, dtype=float)

        best_idx = int(np.argmin(y_obs))
        best_y = float(y_obs[best_idx])
        best_x = X_obs[best_idx]

        acq_opt = self.config.acq_optimizer.lower()

        if acq_opt == "lbfgs":
            # caminho novo: otimiza aquisição em [0,1]^d
            X_new = self._optimize_acq_lbfgs(best_y, q)
        else:
            # caminho existente: sampling + diversidade
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

        return [self._vec_to_param_list(x) for x in np.atleast_2d(X_new)]

    # ---------
    # save/load
    # ---------

    def _get_local_rng_state(self):
        try:
            return list(self.rng.get_state())
        except Exception:
            return None

    def _set_local_rng_state(self, state):
        if state is None:
            return
        try:
            # state[1] pode ter vindo como lista → converte para ndarray
            if isinstance(state, list) and len(state) >= 2 and isinstance(state[1], list):
                state = list(state)
                state[1] = np.asarray(state[1], dtype=np.uint32)
                state = tuple(state)
            self.rng.set_state(state)
        except Exception:
            pass

    # ------------
    # Update / Run
    # ------------
    def update(self) -> List[Individual]:
        if len(self.history_X) == 0:
            raise RuntimeError("Call initialize() and evaluate() before update().")
        self._fit_gp()
        q = max(1, int(self.config.batch_size))
        new_param_lists = self._propose_batch(q)
        return [Individual(param=plist, fitness_function=self.fitness_function) for plist in new_param_lists]

    def run(self, iterations) -> Individual:
        self.inicio = time.time()

        # Pontos iniciais
        pop0 = self.initialize()
        self.evaluate(pop0)
        self.best = min(self.population, key=lambda p:p.fitness)

        # status + log dos iniciais
        best0 = self.get_best_individual(pop0)
        print(f"\n{len(pop0)} Pontos Iniciais: Melhor Fitness = {best0.fitness:.4g}, Parâmetros: {self.display_parameters(best0)}")
        # kernel ainda não existe aqui; loga apenas o que houver
        try:
            self.add_log(0, pop0)
        except Exception:
            pass

        # Iterações BO
        for it in range(1, int(iterations) + 1):

            # atualiza GP e propõe batch
            new_pop = self.update()
            # avalia batch
            self.evaluate(new_pop)
            # anexa diagnósticos do kernel (length scales etc.) no data de cada indivíduo
            diag = self._kernel_diagnostics()
            for ind in new_pop:
                ind.data = ind.data or {}
                ind.data.update(diag)

            self.population.extend(new_pop)
            self.best = min(self.populations, key=lambda p: p.fitness)

            # status desta iteração
            best_new = self.get_best_individual(new_pop)
            pred_mu = best_new.data.get("pred_mu", None)
            pred_sig = best_new.data.get("pred_sigma", None)
            if pred_mu is not None:
                print(f"Ponto {it}: Fitness = {best_new.fitness:.4g} | Previsto = {pred_mu:.4g} ± {pred_sig:.4g}, Parâmetros: {self.display_parameters(best_new)}")
            else:
                print(f"Ponto {it}: Fitness = {best_new.fitness:.4g}, Parâmetros: {self.display_parameters(best_new)}")

            # logging por iteração (incluindo pred e diag do kernel)
            try:
                self.add_log(it, new_pop)
            except Exception:
                pass

        print(f"\nMelhor solução encontrada: Fitness = {self.best.fitness:.4g}, Parâmetros: {self.display_parameters(self.best)}")
        return self.best


# alias
BayesianOptimization = BO
