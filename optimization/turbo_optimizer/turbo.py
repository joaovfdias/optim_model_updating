from ..optimizer import Optimizer
from ..individual import Individual

from typing import Optional
from datetime import datetime
import numpy as np
import math
import csv
import time

from skopt.space import Real
from skopt import gp_minimize

from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class TuRBO(Optimizer):
    def __init__(self, fitness_function, parameters, initial_points=None):
        """
        Otimização Bayesiana utilizando a função "gp_minimize" da biblioteca "scikit-optimize"
        :param fitness_function: função objetivo a ser otimizada (função que recebe lista de valores dos parâmetros e retorna: fitness, [dados]
        :param parameters: lista de objetos da classe Parameters declarados com limites inferior e superior e nome
        :param initial_points: equivalente a "n_initial_points" na função "gp_minimize", referente ao número de avaliações executadas antes da aproximação da função com "base_estimator"
        """
        super().__init__(fitness_function, parameters, population_size=initial_points)

        self.status = False
        self.log = True

        self.populations = []

        self.initial_evaluations = initial_points or 5 * len(parameters)
        self.search_space = [Real(p.lower_bound, p.upper_bound, name=p.key) for p in parameters]

        self.sampling_method = 'sobol'
        self.sampling_methods = {
            "random": self.random_initial_population,
            "lhs": self.LHS_initial_population,
            "sobol": self.sobol_initial_population
        }
        self.bounds = None
        self.build_bounds()

    def build_bounds(self, device=None, dtype=torch.double):
        """
        Creates BoTorch-compatible bounds tensor of shape (2, d)
        from a list of skopt.space.Real objects.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lb = [dim.low for dim in self.search_space]
        ub = [dim.high for dim in self.search_space]

        self.bounds = torch.tensor(
            [lb, ub],
            dtype=dtype,
            device=device
        )

    def set_sampling_method(self, sampling_method):
        if sampling_method in self.sampling_methods:
            self.sampling_method = sampling_method
        else:
            print(
                f"Método de amostragem '{sampling_method}' inválido. Tipos válidos: {list(self.sampling_methods)}")
            return

    def sobol_initial_population(self):

        dim = len(self.parameters)

        sobol = SobolEngine(dim, scramble=True)

        samples = sobol.draw(self.initial_evaluations).numpy()

        lower = np.array([p.lower_bound for p in self.parameters])
        upper = np.array([p.upper_bound for p in self.parameters])

        scaled = lower + samples * (upper - lower)

        pop = [
            Individual(list(scaled[i]), self.fitness_function)
            for i in range(self.initial_evaluations)
        ]

        return pop

    def evaluate_model(self, params):
        self.populations.append(Individual(params, self.fitness_function))
        self.populations[-1].evaluate()
        fitness = self.populations[-1].fitness
        self.best = min(self.populations, key=lambda p:p.fitness)

        if self.status:
            it = max((len(self.populations) - self.initial_evaluations), 0) # mantém IT=0 para avaliações iniciais printadas, começa a contar quando GP assume
            if it > 0:
                print(f"Avaliação {it}: Global Best = {self.best.fitness}, Fitness = {self.populations[-1].fitness}, Parâmetros: {self.display_parameters(self.populations[-1])}")
            else: # altera a mensagem caso esteja nos pontos iniciais ainda
                print(f"Avaliação Inicial {len(self.populations)}: Global Best = {self.best.fitness}, Fitness = {self.populations[-1].fitness}, Parâmetros: {self.display_parameters(self.populations[-1])}")

        if self.log:
            it = max((len(self.populations) - self.initial_evaluations), 0)
            self.add_log(it, [self.populations[-1]])

        return fitness

    def eval_objective(self, x: torch.Tensor) -> torch.Tensor:
        """
        TuRBO tutorial style:
          - input: x is a single point, shape (d,)
          - output: scalar tensor (0-dim), on x.device and x.dtype
        """

        x_raw = unnormalize(x, self.get_bounds())  # self.bounds must be shape (2, d) tensor

        # Treat objective as black-box: no autograd graph
        with torch.no_grad():
            params = x_raw.detach().tolist()
            fitness = self.evaluate_model(params)
            return torch.tensor(fitness, device=x.device, dtype=x.dtype)

    @dataclass
    class TurboState:
        """Classe para armazenar o estado do algoritmo TurBO e atualizar para a busca"""
        dim: int
        batch_size: int
        length: float = 0.8
        length_min: float = 0.5 ** 7
        length_max: float = 1.6
        failure_counter: int = 0
        failure_tolerance: int = float("nan")  # Note: Post-initialized
        success_counter: int = 0
        success_tolerance: int = 10  # Note: The original paper uses 3
        best_value: float = -float("inf")
        restart_triggered: bool = False

    @staticmethod
    def update_state(state: TurboState, Y_next:torch.Tensor) -> TurboState:
        """Atualiza o estado  do TurBO para a busca"""
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

    @staticmethod
    def generate_batch(
            state: TurboState,
            model: SingleTaskGP,  # GP model
            X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
            Y: torch.Tensor,  # Function values
            batch_size: int,
            n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
            num_restarts: int = 10,
            raw_samples: int = 512,
            acqf: str = "ts",  # "ei" or "ts"
    ) -> torch.Tensor:
        """Generate a new batch of points."""
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

    @staticmethod
    def _fit_gp(X, Y):

        likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-8, 1e-3)
        )

        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=X.shape[-1])
        )

        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mll)

        return model

    def population_to_tensor(
            self,
            pop,
            dtype: torch.dtype = torch.double,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.Tensor:
        """
        Convert a list of Individuals/Particles to
        a normalized torch tensor (n, d) in [0,1]^d.
        """

        device = torch.device(device)

        # ---- 1) Extract raw parameter matrix (n, d)
        X_raw = np.array([
            ind.param  # <-- adjust if attribute name differs
            for ind in pop
        ], dtype=float)

        # ---- 2) Build lower/upper bounds arrays
        lb = np.array([p.lower_bound for p in self.parameters], dtype=float)
        ub = np.array([p.upper_bound for p in self.parameters], dtype=float)

        # ---- 3) Normalize to [0,1]^d
        X_unit = (X_raw - lb) / (ub - lb)

        # ---- 4) Convert to torch tensor
        X_tensor = torch.as_tensor(X_unit, dtype=dtype, device=device)

        # Safety clamp
        X_tensor = X_tensor.clamp(0.0, 1.0)

        return X_tensor

    def get_initial_points(
            self,
            dtype: torch.dtype = torch.double,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.Tensor:
        pop = self.initial_population()
        return self.population_to_tensor(pop, dtype=dtype, device=device)


    # Otimizar com scikit-optimize
    def run(self, evaluations: int, acqf: str = "ts", status: bool = True, log: bool = True,
            batch_size: int = 4, n_init: Optional[int] = None, seed: int = 0):
        """
        TuRBO-1 loop like BoTorch tutorial.
        - evaluations: total evaluation budget
        - acqf: "ts" or "ei"
        - batch_size: q
        - n_init: Sobol initial points (default 2*dim, like tutorial)
        """
        assert acqf in ("ts", "ei")

        self.inicio = time.time()
        # self.status = status
        self.log = log

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.logfilename = self.logfilename or f"TuRBO_{timestamp}_acqf={acqf}_q={batch_size}"

        device = self.bounds.device
        dtype = self.bounds.dtype

        dim = self.bounds.shape[1]
        n_init = n_init or (2 * dim)

        # --- 1) initial design in [0,1]^d
        X = self.get_initial_points(dtype=dtype, device=device)

        # --- 2) evaluate initial points (convert to real domain inside eval)
        # We store Y = -fitness, since we minimize fitness but TuRBO maximizes Y
        Y_list = []
        for i in range(X.shape[0]):
            x_raw = unnormalize(X[i], self.bounds)
            with torch.no_grad():
                fitness = float(self.evaluate_model(x_raw.detach().tolist()))
            Y_list.append(-fitness)

        Y = torch.tensor(Y_list, dtype=dtype, device=device).unsqueeze(-1)  # (n_init, 1)

        # tracking TuRBO state
        state = self.TurboState(dim=dim, batch_size=batch_size)

        # reporting
        if status:
            best_fitness = -Y.max().item()
            print(f"[init] n={n_init} | best fitness={best_fitness:.6g} | TR length={state.length:.3g}")

        # --- 3) TuRBO iterations
        n_evals = n_init
        while n_evals < evaluations and not state.restart_triggered:
            # Fit GP on (X,Y)
            model = self._fit_gp(X, Y)

            # Propose batch in [0,1]^d
            X_next = self.generate_batch(
                state=state,
                model=model,
                X=X,
                Y=Y,
                batch_size=batch_size,
                acqf=acqf,
            )

            # Evaluate batch
            Y_next_list = []
            for j in range(X_next.shape[0]):
                x_raw = unnormalize(X_next[j], self.bounds)
                with torch.no_grad():
                    fitness = float(self.evaluate_model(x_raw.detach().tolist()))
                Y_next_list.append(-fitness)

            Y_next = torch.tensor(Y_next_list, dtype=dtype, device=device).unsqueeze(-1)  # (q,1)

            # Append data
            X = torch.cat([X, X_next], dim=0)
            Y = torch.cat([Y, Y_next], dim=0)
            n_evals += X_next.shape[0]

            # Update TR state (based on new Y)
            state = self.update_state(state, Y_next)

            if status:
                best_fitness = -Y.max().item()
                print(f"[eval {n_evals:4d}/{evaluations}] best fitness={best_fitness:.6g} | "
                      f"TR length={state.length:.3g} | restart={state.restart_triggered}")

            # # optional logging hook (keep your existing system)
            # if self.log:
            #     it = max((len(self.populations) - self.initial_evaluations), 0)
            #     self.add_log(it, [self.populations[-1]])

        fim = time.time()
        if self.log:
            self.log_time(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        best_individual = self.get_best_individual(self.populations)
        print(f"\nMelhor solução encontrada: Fitness = {best_individual.fitness}, "
              f"Parâmetros: {self.display_parameters(best_individual)}")

        return {
            "X": X,  # normalized (n, d)
            "Y": Y,  # objective values (maximize) (n,1) where Y=-fitness
            "best_fitness": -Y.max().item(),
            "state": state,
        }


    def add_log_specs(self, specs_dictionary, length_scales):
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')

            # Adiciona uma linha vazia
            writer.writerow([])

            # Adiciona o título "Specifications"
            writer.writerow(["Specifications"])

            # Adiciona as keys em uma linha
            writer.writerow(specs_dictionary.keys())

            # Adiciona os valores correspondentes em outra linha
            writer.writerow(specs_dictionary.values())

            # comprimentos de escala e sensibilidade
            writer.writerow([])
            writer.writerow(["Length scales:"] + [length_scales])

            sensitivities = 1 / np.array(length_scales)
            relative_sensitivities = sensitivities / np.sum(sensitivities)

            writer.writerow(["Relative sensitivities:"] + [relative_sensitivities])
