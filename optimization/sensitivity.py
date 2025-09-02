
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Sequence, Any
import json, os
import numpy as np

from scipy.stats import spearmanr


# def _rankdata(a: np.ndarray) -> np.ndarray:
#     a = np.asarray(a, dtype=float)
#     n = a.size
#     order = a.argsort(kind="mergesort")
#     ranks = np.empty(n, dtype=float)
#     ranks[order] = np.arange(1, n + 1, dtype=float)
#     # corrigir empates: média das posições
#     # detecta blocos de valores iguais no array ordenado
#     vals = a[order]
#     i = 0
#     while i < n - 1:
#         j = i
#         while j + 1 < n and vals[j + 1] == vals[i]:
#             j += 1
#         if j > i:
#             mean_rank = (ranks[order][i:j+1].mean())
#             ranks[order][i:j+1] = mean_rank
#         i = j + 1
#     return ranks

def _spearmanr_1d(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Spearman via SciPy.
    """
    rho, p = spearmanr(x, y, nan_policy="omit")
    if rho is None or np.isnan(rho) or np.isnan(p):
        return 0.0, 1.0
    return float(rho), float(p)

    # import math
    # x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)
    # n = x.size
    # if n != y.size or n < 3:
    #     return np.nan, np.nan
    # if np.all(x == x[0]) or np.all(y == y[0]):
    #     return 0.0, 1.0
    # rx = _rankdata(x)
    # ry = _rankdata(y)
    # rx = (rx - rx.mean()) / (rx.std() or 1.0)
    # ry = (ry - ry.mean()) / (ry.std() or 1.0)
    # rho = float(np.clip((rx @ ry) / (n - 1), -1.0, 1.0))
    # t = rho * np.sqrt((n - 2) / (1.0 - rho * rho + 1e-12))
    # z = abs(t)
    # Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # p = 2 * (1 - Phi)
    # return rho, float(p)


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class SensitivityResult:
    names: List[str]
    rho: List[float]
    pval: List[float]
    selected_idx: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {"names": self.names, "rho": self.rho, "pval": self.pval, "selected_idx": self.selected_idx}

class SensitivityAnalyzer:
    """
    Análise de sensibilidade baseada em Spearman |rho| entre parâmetros e fitness.
    """

    def __init__(self, param_names: List[str], X: np.ndarray, y: np.ndarray):
        """
        X: shape (N, D) amostras de parâmetros
        y: shape (N,) fitness correspondente
        """
        self.param_names = list(param_names)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

        ok = np.isfinite(self.X).all(axis=1) & np.isfinite(self.y)
        self.X = self.X[ok]
        self.y = self.y[ok]

    @classmethod
    def from_optimizer(cls, opt) -> "SensitivityAnalyzer":
        """
        Coleta (X, y) do otimizador:
        - BO: usa history_X, history_y
        - GA/PSO: concatena todas as populações avaliadas (fitness não-None)
        """
        # nomes dos parâmetros
        names = [getattr(p, "key", f"p{i}") for i, p in enumerate(opt.parameters)]

        # BO
        if opt.__class__.__name__ == "BO" or hasattr(opt, "history_X"):
            HX = getattr(opt, "history_X", [])
            Hy = getattr(opt, "history_y", [])
            X = np.array(HX, dtype=float)
            y = np.array(Hy, dtype=float)
            return cls(names, X, y)

        # GA/PSO
        rows = []
        ys = []
        for pop in getattr(opt, "populations", []):
            for ind in pop:
                if ind.fitness is None:  # pular não avaliados
                    continue
                rows.append(np.array(ind.param, dtype=float))
                ys.append(float(ind.fitness))
        if not rows:
            raise ValueError("Nenhum indivíduo avaliado encontrado em opt.populations.")
        X = np.vstack(rows)
        y = np.array(ys, dtype=float)
        return cls(names, X, y)

    def compute(self, absolute: bool = True) -> Tuple[List[float], List[float]]:
        """Retorna (rho, pval) por parâmetro. Se absolute=True usa |rho| para ranking."""
        D = self.X.shape[1]
        rho, pval = [], []
        for j in range(D):
            r, p = _spearmanr_1d(self.X[:, j], self.y)
            rho.append(abs(r) if absolute else r)
            pval.append(p)
        return rho, pval

    def select(self,
               rho: List[float],
               pval: List[float],
               mode: str = "threshold",
               threshold: float = 0.3,
               top_k: Optional[int] = None,
               max_pval: Optional[float] = None) -> List[int]:
        """
        Critério de seleção:
          - mode='threshold': escolhe índices com |rho| >= threshold e (opcional) pval <= max_pval
          - mode='topk': escolhe os 'top_k' maiores |rho| (aplica max_pval se dado)
        """
        idx = list(range(len(rho)))
        if mode == "threshold":
            sel = [i for i in idx if rho[i] >= threshold and (max_pval is None or pval[i] <= max_pval)]
            return sel
        elif mode == "topk":
            if top_k is None or top_k <= 0:
                raise ValueError("top_k deve ser > 0 para mode='topk'.")
            order = sorted(idx, key=lambda i: rho[i], reverse=True)
            ordered = [i for i in order if (max_pval is None or pval[i] <= max_pval)]
            return ordered[:top_k]
        else:
            raise ValueError("mode deve ser 'threshold' ou 'topk'.")

    def run(self,
            mode: str = "threshold",
            threshold: float = 0.3,
            top_k: Optional[int] = None,
            max_pval: Optional[float] = 0.05,
            absolute: bool = True,
            verbose: bool = True) -> SensitivityResult:
        rho, pval = self.compute(absolute=absolute)
        sel = self.select(rho, pval, mode=mode, threshold=threshold, top_k=top_k, max_pval=max_pval)

        if verbose:
            print("[sensitivity] Spearman |rho| por parâmetro:")
            for name, r, p in zip(self.param_names, rho, pval):
                star = " *" if (name in [self.param_names[i] for i in sel]) else ""
                sig = f"(p={p:.3g})"
                print(f"  - {name:>15s}: {r:.4f} {sig}{star}")
            if mode == "threshold":
                print(f"[sensitivity] Seleção: |rho| >= {threshold} e p <= {max_pval}")
            else:
                print(f"[sensitivity] Seleção: top_k = {top_k} (p <= {max_pval} se fornecido)")
            print(f"[sensitivity] Escolhidos: {[self.param_names[i] for i in sel]}")

        return SensitivityResult(self.param_names, rho, pval, sel)


    def export_scores_csv(self, res: SensitivityResult, filename: str):
        _ensure_dir(filename)
        import csv
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["param", "spearman_abs_rho", "p_value", "selected"])
            for i, name in enumerate(res.names):
                w.writerow([name, f"{res.rho[i]:.6g}", f"{res.pval[i]:.6g}", int(i in res.selected_idx)])
        print(f"[sensitivity] CSV salvo em: {filename}")

    def export_selection_json(self,
                              res: SensitivityResult,
                              all_parameters: Sequence[Any],
                              filename: str):
        """
        Gera um JSON com:
          - lista completa de scores
          - e um bloco 'selected_parameters' com os Parameters (kind, key, bounds) para recriar depois.
        """
        _ensure_dir(filename)

        def param_to_dict(p):
            d = {"kind": p.__class__.__name__, "key": getattr(p, "key", None)}
            if hasattr(p, "lower_bound"): d["lower"] = float(p.lower_bound)
            if hasattr(p, "upper_bound"): d["upper"] = float(p.upper_bound)
            return d

        payload = {
            "version": 1,
            "method": "spearman",
            "names": res.names,
            "spearman_abs_rho": res.rho,
            "p_values": res.pval,
            "selected_idx": res.selected_idx,
            "selected_parameters": [param_to_dict(all_parameters[i]) for i in res.selected_idx],
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[sensitivity] Seleção JSON salva em: {filename}")

    def build_parameter_subset(self, all_parameters: Sequence[Any], selected_idx: List[int]) -> List[Any]:
        """Retorna a sublista de Parameters a calibrar."""
        return [all_parameters[i] for i in selected_idx]
