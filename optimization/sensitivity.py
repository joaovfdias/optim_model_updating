from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
from datetime import datetime


# ParamSpec + Amostragem (DoE)

@dataclass
class ParamSpec:
    name: str
    lower: float
    upper: float
    kind: Literal["continuous", "integer"] = "continuous"


class Sampler:
    @staticmethod
    def random(n: int, specs: Sequence[ParamSpec], rng: np.random.Generator) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for _ in range(n):
            row = {}
            for p in specs:
                u = rng.random()
                v = p.lower + u * (p.upper - p.lower)
                row[p.name] = int(round(v)) if p.kind == "integer" else float(v)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def lhs(n: int, specs: Sequence[ParamSpec], rng: np.random.Generator) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n deve ser positivo")
        cols: Dict[str, np.ndarray] = {}
        for p in specs:
            cut = np.linspace(0.0, 1.0, n + 1)
            u = rng.random(n)
            pts = cut[:-1] + u * (1.0 / n)
            rng.shuffle(pts)
            vals = p.lower + pts * (p.upper - p.lower)
            vals = np.rint(vals).astype(int) if p.kind == "integer" else vals.astype(float)
            cols[p.name] = vals
        return pd.DataFrame(cols)


# Detecção de métricas do LOG

_SERVICE_COLS = {"Iteration", "Individual", "Fitness", "Global Best", "Time (s)"}

def _is_service_col(c: str) -> bool:
    return c in _SERVICE_COLS or c.startswith("pso.v")  # velocidades no PSO


def _default_metric_detector(c: str) -> bool:
    """
    Compatível com cabeçalhos gerados pelo teu Optimizer:
    - vetores: "freq #1", "MAC #1", etc.
    - permite variações "freq_1", "MAC_1", "metric_*"
    """
    if _is_service_col(c):
        return False
    c0 = c.strip()
    return bool(
        re.match(r'^(freq|Freq|mac|MAC|metric_)\b', c0) or
        re.search(r'\s#\d+$', c0)  # ex.: "freq #1", "MAC #2", "modos #3"
    )


# Cálculo de correlações

def compute_spearman_multi(
    df: pd.DataFrame,
    *,
    param_cols: Optional[List[str]] = None,
    metric_cols: Optional[List[str]] = None,
    fitness_col: Optional[str] = None,
    minimize: bool = True,
    is_metric_fn: Callable[[str], bool] = _default_metric_detector,
    param_keys_hint: Optional[List[str]] = None,  # ex.: [p.key for p in optimizer.parameters]
    drop_extra_cols: Iterable[str] = (),
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Retorna:
      - corr_df: DataFrame (index=parâmetros, columns=métricas) com ρ (com sinal)
      - overall: Series  (index=parâmetros) com ρ vs fitness (se fitness_col fornecida)
    """
    df = df.copy()

    # Param cols
    if param_cols is None:
        if param_keys_hint:
            param_cols = [c for c in param_keys_hint if c in df.columns]
        else:
            blacklist = set(drop_extra_cols) | _SERVICE_COLS
            if fitness_col:
                blacklist.add(fitness_col)
            # métricas detectadas automaticamente
            metrics_auto = [c for c in df.columns if is_metric_fn(c)]
            blacklist.update(metrics_auto)
            # tudo que sobrar (numérico) vira candidato a parâmetro
            param_cols = [c for c in df.columns if c not in blacklist]

    # Metric cols
    if metric_cols is None:
        metric_cols = [c for c in df.columns if is_metric_fn(c)]
        if not metric_cols:
            raise ValueError("Nenhuma coluna de métrica detectada; passe metric_cols explicitamente.")

    # sanity
    miss_p = [c for c in param_cols if c not in df.columns]
    miss_m = [c for c in metric_cols if c not in df.columns]
    if miss_p or miss_m:
        raise ValueError(f"Colunas ausentes. params={miss_p}, metrics={miss_m}")

    # Spearman por métrica
    rows = []
    for p in param_cols:
        x = pd.to_numeric(df[p], errors="coerce")
        row: Dict[str, float] = {}
        for m in metric_cols:
            y = pd.to_numeric(df[m], errors="coerce")
            mask = ~(x.isna() | y.isna())
            if mask.sum() < 3:
                rho = np.nan
            else:
                rho, _ = spearmanr(x[mask], y[mask])
            row[m] = float(rho) if rho is not None else np.nan
        rows.append(pd.Series(row, name=p))
    corr_df = pd.DataFrame(rows)

    # Spearman com fitness geral (opcional)
    overall = None
    if fitness_col and fitness_col in df.columns:
        f = pd.to_numeric(df[fitness_col], errors="coerce")
        f_eff = -f if minimize else f
        vals: Dict[str, float] = {}
        for p in param_cols:
            x = pd.to_numeric(df[p], errors="coerce")
            mask = ~(x.isna() | f_eff.isna())
            if mask.sum() < 3:
                rho = np.nan
            else:
                rho, _ = spearmanr(x[mask], f_eff[mask])
            vals[p] = float(rho) if rho is not None else np.nan
        overall = pd.Series(vals, name="rho_fitness").sort_values(ascending=False)

    return corr_df, overall


def plot_heatmap_corr(corr_df: pd.DataFrame, title: str = "Spearman (parâmetros × métricas)", auto_save: bool = True):
    if corr_df.empty:
        raise ValueError("corr_df vazio")
    fig_w = max(6, 0.7 * corr_df.shape[1])
    fig_h = max(4, 0.45 * corr_df.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr_df.values, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr_df.shape[0]))
    ax.set_yticklabels(corr_df.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ρ (Spearman)")
    ax.set_title(title)
    ax.set_xlabel("Métricas")
    ax.set_ylabel("Parâmetros")
    # anota valores se não for gigante
    if corr_df.shape[0] * corr_df.shape[1] <= 200:
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                v = corr_df.iat[i, j]
                if not (isinstance(v, float) and math.isnan(v)):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()

    if auto_save:
        out_dir = os.path.join(os.getcwd(), "plot")
        os.makedirs(out_dir, exist_ok=True)

        # gera nome do arquivo com timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"heatmap_sensitivity_{ts}.png"
        fpath = os.path.join(out_dir, fname)

        plt.savefig(fpath, dpi=300)
        print(f"[OK] Heatmap salvo em {fpath}")

    plt.show()


def recommend_cut(
    corr_df: pd.DataFrame,
    *,
    strategy: Literal["max_abs", "mean_abs"] = "max_abs",
    tau: Optional[float] = None,
    topk: Optional[int] = None,
) -> Tuple[List[str], pd.DataFrame]:
    if corr_df.empty:
        return [], pd.DataFrame(columns=["score"])
    abs_df = corr_df.abs()
    score = abs_df.max(axis=1) if strategy == "max_abs" else abs_df.mean(axis=1)
    if topk is not None:
        keep = list(score.nlargest(topk).index)
    else:
        if tau is None:
            with np.errstate(invalid="ignore"):
                tau = float(np.nanquantile(score.values, 0.7))  # heurística ~top 30%
        keep = list(score[score >= float(tau)].index)
    ranking = pd.DataFrame({"score": score}).sort_values("score", ascending=False)
    return keep, ranking


# Orquestrador

class SensitivityAnalyzer:
    def __init__(self, *, minimize: bool = True, is_metric_fn: Callable[[str], bool] = _default_metric_detector):
        self.minimize = minimize
        self.is_metric_fn = is_metric_fn

    # (1) Histórico / DoE já avaliados
    def from_history(
        self,
        df: pd.DataFrame,
        *,
        param_cols: Optional[List[str]] = None,
        metric_cols: Optional[List[str]] = None,
        fitness_col: Optional[str] = None,
        param_keys_hint: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        return compute_spearman_multi(
            df,
            param_cols=param_cols,
            metric_cols=metric_cols,
            fitness_col=fitness_col,
            minimize=self.minimize,
            is_metric_fn=self.is_metric_fn,
            param_keys_hint=param_keys_hint,
        )

    # (2) Gera DoE a partir do teu Optimizer.parameters
    def from_optimizer_doe(
        self,
        optimizer,
        evaluate_fn: Callable[[pd.Series], Union[float, Dict[str, float]]],
        *,
        n: Optional[int] = None,
        sampler: Literal["lhs", "random"] = "lhs",
        seed: Optional[int] = None,
        fitness_col: str = "Fitness",
        metric_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        specs: List[ParamSpec] = []
        for p in optimizer.parameters:
            # compatível com .key, .lower_bound, .upper_bound do teu código
            name = p.key
            lb = float(p.lower_bound)
            ub = float(p.upper_bound)
            kind = getattr(p, "kind", "continuous")
            specs.append(ParamSpec(name, lb, ub, kind))
        n = n or 10 * max(1, len(specs))

        rng = np.random.default_rng(seed)
        X = Sampler.lhs(n, specs, rng) if sampler == "lhs" else Sampler.random(n, specs, rng)

        results: List[Dict[str, float]] = []
        for _, r in X.iterrows():
            out = evaluate_fn(r)
            if isinstance(out, dict):
                results.append(out)
            else:
                results.append({fitness_col: float(out)})
        Y = pd.DataFrame(results)

        df_eval = pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)

        corr_df, overall = compute_spearman_multi(
            df_eval,
            param_cols=[s.name for s in specs],
            metric_cols=metric_cols,  # se None, detecta automaticamente
            fitness_col=fitness_col if fitness_col in df_eval.columns else None,
            minimize=self.minimize,
            is_metric_fn=self.is_metric_fn,
        )
        return corr_df, overall, df_eval

    # (3) Workflow completo com prompt opcional
    def workflow(
        self,
        df: pd.DataFrame,
        *,
        fitness_col: Optional[str] = None,
        param_cols: Optional[List[str]] = None,
        metric_cols: Optional[List[str]] = None,
        param_keys_hint: Optional[List[str]] = None,
        strategy: Literal["max_abs", "mean_abs"] = "max_abs",
        tau: Optional[float] = None,
        topk: Optional[int] = None,
        show_plot: bool = True,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
    ) -> List[str]:
        corr_df, overall = self.from_history(
            df,
            param_cols=param_cols,
            metric_cols=metric_cols,
            fitness_col=fitness_col,
            param_keys_hint=param_keys_hint,
        )

        print("\n=== Correlação por métrica (ρ de Spearman) ===")
        print(corr_df.round(3).to_string())

        if show_plot:
            try:
                plot_heatmap_corr(corr_df, title="Spearman por métrica (parâmetros × métricas)")
            except Exception as e:
                print(f"[Aviso] Falha ao exibir heatmap: {e}")

        if overall is not None:
            print("\n=== Correlação com fitness geral (ρ; ajustado se minimização) ===")
            print(overall.round(3).to_string())

        suggested, ranking = recommend_cut(corr_df, strategy=strategy, tau=tau, topk=topk)

        print("\n=== Ranking por importância (|ρ| → score) ===")
        print(ranking.round(3).to_string())

        all_params = list(corr_df.index)
        print("\n=== Parâmetros ===")
        for i, p in enumerate(all_params, start=1):
            print(f"{i}. {p}")
        print("0. All")
        print(f"\nSugerido manter: {suggested}")

        if not interactive:
            print("[Info] Modo não interativo: usando sugestão automática.")
            return suggested

        user = input_fn(
            "\n[ Sensibilidade ]\n"
            "ENTER para aceitar a sugestão\n"
            "ou digite índices (ex.: 1,3,5) ou '0' para All: "
        ).strip()

        if user == "":
            chosen = suggested
        elif user == "0":
            chosen = all_params
        else:
            try:
                idxs = [int(s) for s in user.replace(" ", "").split(",") if s]
                chosen = [all_params[i - 1] for i in idxs if 1 <= i <= len(all_params)]
                if not chosen:
                    print("[Aviso] Nenhum índice válido. Usando sugestão.")
                    chosen = suggested
            except Exception:
                print("[Aviso] Entrada inválida. Usando sugestão.")
                chosen = suggested

        print(f"\n[OK] Parâmetros selecionados: {chosen}")
        return chosen
