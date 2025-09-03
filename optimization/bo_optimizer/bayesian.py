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
    alpha: float = 1e-6 # ruído/regularização, aumentar para altos níveis de ruído
    normalize_y: bool = True
    n_restarts_optimizer: int = 5

    acquisition: str = "EI"
    xi: float = 0.01 # para EI e POI
    kappa: float = 2.576 # para UCB

    init_points: Optional[int] = None # computado depois conforme número de parâmetros
    batch_size: int = 1 # avaliações por iteração
    max_iter: Optional[int] = 100
    random_state: Optional[int] = None # reprodutibilidade

    n_acq_candidates: int = 5000 # candidatos amostrados pelo GP para otimizar a acq_func
    n_local_perturb: int = 800 # amostras locais ao redor do melhor ponto observado

    minimize: bool = True

    _PRESETS = { # configurações que afetam o custo do BO
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
    def from_preset(cls, name: str, n_dims: Optional[int] = None, base: Optional["BOConfig"] = None, overwrite: bool = False) -> "BOConfig":
        """
        Cria um BOConfig a partir de um preset (função interna).
        - name: nome do preset em _PRESETS
        - n_dims: calcula init_points via init_multiplier
        - base: serve como ponto de partida (os campos do preset serão aplicados sobre ele)
        - overwrite: se True: preset sobrescreve campos do base. Se False: preserva campos que o usuário já alterou no base (comparando com o default).
        """
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls._PRESETS.keys())}")

        preset = cls._PRESETS[name].copy()
        # calcula init_points a partir de init_multiplier se for o caso
        preset_init = None
        if n_dims is not None and "init_multiplier" in preset:
            preset_init = max(preset.get("min_init", 8), int(np.ceil(preset["init_multiplier"] * n_dims)))

        default = cls()              # valores padrão do dataclass
        cfg = cls() if base is None else cls(**asdict(base))

        # campos explícitos do preset a aplicar (ignorar keys de controle)
        keys_to_apply = {k: v for k, v in preset.items() if k not in ("init_multiplier", "min_init")}

        for k, v in keys_to_apply.items():
            if not hasattr(cfg, k):
                continue
            if not overwrite and base is not None:
                # se o usuário mudou esse campo (base != default), manter
                if getattr(base, k) != getattr(default, k):
                    continue
            setattr(cfg, k, v)

        if preset_init is not None:
            if overwrite:
                cfg.init_points = preset_init
                # se base foi fornecido e usuário já alterou init_points, manter
            else:
                if base is None or getattr(base, "init_points") == getattr(default, "init_points"):
                    cfg.init_points = preset_init
                # caso contrário, valor do base

        return cfg

    def apply_preset(self, name: str, n_dims: Optional[int] = None, overwrite: bool = False) -> None:
        """
        Aplica o preset *na própria instância*, preservando campos conforme overwrite.
        """
        new_cfg = self.from_preset(name, n_dims=n_dims, base=self, overwrite=overwrite)
        # atualiza a instância em-place
        for k, v in asdict(new_cfg).items():
            setattr(self, k, v)

    # def to_json(self, path: str) -> None:
    #     d = asdict(self)
    #     if d.get("kernel") is not None:
    #         d["kernel"] = None
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(d, f, indent=2)
    #
    # @classmethod
    # def from_json(cls, path: str) -> "BOConfig":
    #     with open(path, "r", encoding="utf-8") as f:
    #         d = json.load(f)
    #     return cls(**d)


class BO(Optimizer):
    def __init__(
        self,
        fitness_function: Callable[[List[float]], Any],
        parameters: Sequence[Parameter],
        population_size: int = 1,
        config: Optional[BOConfig] = None,
):
        super().__init__(fitness_function, list(parameters), population_size)

        self.algorithms[self.__class__.__name__] = Individual

        self.config = config or BOConfig()
        self.rng = np.random.RandomState(self.config.random_state)

        self.history_X: List[np.ndarray] = []
        self.history_y: List[float] = []
        self.gp: Optional[GaussianProcessRegressor] = None

        self.parameters = list(parameters)
        self.n_dims = len(self.parameters)

        # compute default init_points if not provided
        if self.config.init_points is None:
            self.config.init_points = self.config.computed_init_points(self.n_dims, multiplier=3.0, min_points=8)

        self.bounds = np.array([[getattr(p, "lower_bound"), getattr(p, "upper_bound")] for p in self.parameters], dtype=float)
        self.population: List[Individual] = []

    def _default_kernel(self, y):
        span = (self.bounds[:, 1] - self.bounds[:, 0])
        # chute inicial ~20% da faixa por dimensão
        ls0 = np.maximum(1e-12, 0.2 * span)
        # bounds por dimensão (3 ordens abaixo/acima da escala)
        ls_bounds = [(max(1e-12, 1e-3 * s), 1e3 * s) for s in span]

        var_y = float(np.var(y)) if y.size > 1 else 1.0
        nl_bounds = (max(1e-12, 1e-9 * var_y), max(1e-9, 1e2 * var_y))

        base = Matern(length_scale=ls0, length_scale_bounds=ls_bounds, nu=2.5)
        kernel = C(1.0, (1e-6, 1e6)) * base + WhiteKernel(noise_level=1e-6, noise_level_bounds=nl_bounds)
        return kernel

    def _vec_to_param_list(self, x: np.ndarray) -> List[float]:
        return [float(v) for v in x]

    def _param_list_to_vec(self, param_list: Sequence[float]) -> np.ndarray:
        return np.asarray(param_list, dtype=float)


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

        # ensure Individuals
        # pop_checked: List[Individual] = []

        for p in pop:
            if not isinstance(p, Individual):
                raise TypeError("Initial_population must return a list of 'Individual' objects")

        #     if isinstance(p, Individual):
        #         pop_checked.append(p)
        #     elif isinstance(p, dict):
        #         pop_checked.append(Individual(param=p, fitness_function=self.fitness_function))
        #     else:
        #         # assume list/sequence of values in same order as parameters
        #         pop_checked.append(Individual(param=self._vec_to_param_list(np.asarray(p)), fitness_function=self.fitness_function))
        # self.population = pop_checked

        self.population = pop
        return pop

    # EVALUATION ----------
    def evaluate(self, population: Optional[List[Individual]] = None):


        pop = population or self.population

        for ind in pop:
            ind.evaluate()
            x = self._param_list_to_vec(ind.param)
            y = float(ind.fitness)

            self.history_X.append(x)
            self.history_y.append(y if self.config.minimize else -y)

            # armazenar previsão e incerteza
            if self.gp is not None:
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                ind.data["pred_fitness"] = float(mu[0])
                ind.data["pred_sigma"] = float(sigma[0])

    def _fit_gp(self):
        X = np.vstack(self.history_X)
        y = np.asarray(self.history_y, dtype=float)

        # kernel inicial: usa kernel_ otimizado anterior se existir
        if self.gp is not None and hasattr(self.gp, "kernel_"):
            kernel0 = self.gp.kernel_
        else:
            kernel0 = self.config.kernel or self._default_kernel(y)

        self.gp = GaussianProcessRegressor(
            kernel=kernel0,
            alpha=self.config.alpha,
            normalize_y=self.config.normalize_y,
            n_restarts_optimizer=max(1, self.config.n_restarts_optimizer),  # garantir >0
            optimizer="fmin_l_bfgs_b",
            random_state=self.rng,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.gp.fit(X, y)

        # print("LML:", getattr(self.gp, "log_marginal_likelihood_value_", None))
        # print("Kernel_:", getattr(self.gp, "kernel_", None))

    def _extract_length_scales(self, kernel):
        # busca recursiva pelo componente que possui length_scale
        if hasattr(kernel, "length_scale"):
            ls = np.atleast_1d(kernel.length_scale).astype(float).tolist()
            return ls
        for attr in ("k1", "k2"):
            if hasattr(kernel, attr):
                ls = self._extract_length_scales(getattr(kernel, attr))
                if ls is not None:
                    return ls
        return None

    # ACQUISITION FUNCTION ----------
    def _predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: # prediz média e desvio padrão do GP
        assert self.gp is not None
        mu, std = self.gp.predict(X, return_std=True)
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

    # CANDIDATOS ----------
    def _sample_candidates(self, n: int) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(n, self.n_dims)) # testar outras amostragens

    def _local_around_best(self, best_x: np.ndarray, n: int) -> np.ndarray:
        span = (self.bounds[:, 1] - self.bounds[:, 0])
        scale = 0.05 * span # testar outras escalas?
        X = best_x + self.rng.randn(n, self.n_dims) * scale
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])

    def _deduplicate(self, X: np.ndarray, existing: np.ndarray, tol: float = 1e-3) -> np.ndarray: # remove pontos próximos aos já observados
        if existing.size == 0:
            return X
        keep = []
        for i in range(X.shape[0]):
            span = self.bounds[:, 1] - self.bounds[:, 0]
            diff = (existing - X[i]) / span  # normaliza por faixa
            d = np.min(np.linalg.norm(diff, axis=1))
            if d > tol: # testar outras tolerâncias?
                keep.append(i)
        if keep:
            return X[keep]
        return X[:0]

    def _propose_batch(self, q: int) -> List[List[float]]:
        assert self.gp is not None and len(self.history_y) > 0
        X_obs = np.vstack(self.history_X)
        y_obs = np.asarray(self.history_y, dtype=float)

        best_idx = int(np.argmin(y_obs)) # melhor ponto até agora
        best_y = float(y_obs[best_idx])
        best_x = X_obs[best_idx]

        Xc = self._sample_candidates(self.config.n_acq_candidates) # gera candidatos aleatórios no espaço inteiro
        Xc = np.vstack([Xc, self._local_around_best(best_x, self.config.n_local_perturb)]) # adiciona candidatos ao redor do melhor atual (para refinamento/exploitation)
        Xc = self._deduplicate(Xc, X_obs) # remove candidatos muito próximos aos já testados
        if Xc.shape[0] == 0:
            Xc = self._sample_candidates(max(self.config.n_acq_candidates // 2, q))

        acq_vals = self._acquisition(Xc, best_y) # recebem score da acq_func, maior = melhor

        span = (self.bounds[:, 1] - self.bounds[:, 0])
        span = np.where(span == 0, 1.0, span)  # evita divisão por zero
        hard_radius = 1e-3  # relativo ao espaço (≈0.1% da diagonal normalizada)
        soft_sigma = 0.02  # largura da penalização (2% ~ suave)

        chosen: List[int] = []
        acq_copy = acq_vals.copy()
        Xc_copy = Xc.copy()
        for _ in range(q):
            idx = int(np.argmax(acq_copy))  # escolhe canditado com melhor valor de aquisição
            chosen.append(idx)

            # distância NORMALIZADA
            diff = (Xc_copy - Xc_copy[idx]) / span
            d = np.linalg.norm(diff, axis=1)

            # remove pontos muito próximos (em termos relativos)
            acq_copy[d < hard_radius] = -np.inf

            # penalização suave: decai com a distância normalizada
            acq_copy -= 0.1 * np.exp(-(d ** 2) / (2 * (soft_sigma ** 2))) # penaliza próximos

        X_new = Xc[chosen]
        return [self._vec_to_param_list(x) for x in X_new]

    # UPDATE / RUN ----------
    def update(self) -> List[Individual]:
        if len(self.history_X) == 0:
            raise RuntimeError("Call initialize() and evaluate() before update().")
        self._fit_gp()
        q = max(1, int(self.config.batch_size))
        new_param_lists = self._propose_batch(q)
        return [Individual(param=plist, fitness_function=self.fitness_function) for plist in new_param_lists]

    def run(self, max_iter=None, status=True, log=True):
        """
        Execução do BO:
          - cria população inicial (n0 = init_points)
          - avalia, registra pop0, atualiza best
          - itera propondo batches (q = batch_size), avalia e registra
          - aplica critério de parada via Optimizer.tolerance entre populações consecutivas
          - registra specs (inclui length_scales) e tempos ao final
        """
        self.inicio = time.time()
        self.status = status
        max_iter = int(max_iter if max_iter is not None else self.config.max_iter)

        # 1) População inicial
        pop0 = self.initialize()
        self.evaluate(pop0)
        # armazena como primeira população
        self.populations = [pop0]

        # best global (usado pelo add_log para BO)
        self.best = self.get_best_individual(pop0)

        if self.status:
            print(f"\n{len(pop0)} Pontos Iniciais: Melhor Fitness = {self.best.fitness:.4g}, "
                  f"Parâmetros: {self.display_parameters(self.best)}")

        # logging inicial
        if log:
            self.add_log(0, pop0)

        # 2) Loop principal
        for it in range(1, max_iter + 1):
            # ajusta GP com t0do historico
            new_pop = self.update()  # propõe 'q' novos pontos
            self.evaluate(new_pop)  # avalia
            self.populations.append(new_pop)  # registra batch como “população” desta iteração

            # atualiza melhor global
            best_new = self.get_best_individual(new_pop)
            if best_new.fitness < self.best.fitness:
                self.best = best_new

            # mensagens de status (inclui previsão do GP se disponível)
            pred = best_new.data.get("pred_fitness", None)
            sig = best_new.data.get("pred_sigma", None)
            if self.status:
                if pred is not None:
                    print(f"Ponto {it}: Fitness real = {best_new.fitness:.4g}, "
                          f"Previsto = {pred:.4g} ± {sig:.4g}, "
                          f"Parâmetros: {self.display_parameters(best_new)}")
                else:
                    print(f"Ponto {it}: Fitness = {best_new.fitness:.4g}, "
                          f"Parâmetros: {self.display_parameters(best_new)}")

            # logging por iteração
            if log:
                self.add_log(it, new_pop)

            # critério de parada (usa as duas últimas “populações”/batches)
            if self.tolerance(self.populations[-2], self.populations[-1]):
                break

        # 3) Finalização
        fim = time.time()

        if log:
            # escreve specs (inclui length_scales e sensibilidades relativas) e tempos
            self.log_specs()
            self.log_time(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        print(f"\nMelhor solução encontrada: Fitness = {self.best.fitness:.4g}, "
              f"Parâmetros: {self.display_parameters(self.best)}")

        return self.best

    @property
    def specs(self):
        cfg = asdict(self.config)
        kernel_str = None
        length_scales = None
        rel_sens = None

        if self.gp is not None and hasattr(self.gp, "kernel_"):
            kernel_str = str(self.gp.kernel_)
            length_scales = self._extract_length_scales(self.gp.kernel_)
            if length_scales:
                arr = np.array(length_scales, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sens = 1.0 / arr
                    if np.all(np.isfinite(sens)) and sens.sum() > 0:
                        rel_sens = (sens / sens.sum()).tolist()

        return {
            "config": cfg,
            "kernel_str": kernel_str,
            "length_scales": length_scales,
            "relative_sensitivities": rel_sens,
        }


# nome alternativo (alias)
BayesianOptimization = BO
