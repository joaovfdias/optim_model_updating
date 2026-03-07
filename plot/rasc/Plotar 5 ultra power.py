from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Estilo global de figura
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.subplot.left']   = 0.10
plt.rcParams['figure.subplot.right']  = 0.80
plt.rcParams['figure.subplot.top']    = 0.90
plt.rcParams['figure.subplot.bottom'] = 0.10


# =========================
# Leitor de CSV (robusto)
# =========================
class CSVLoader:
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[\s_]+", "", s.strip().lower())

    def read(self, path: Path) -> pd.DataFrame:
        # tenta com decimal=',' e depois sem; tolerante a encoding
        for enc in ("utf-8", "latin1"):
            for dec in (",", "."):
                try:
                    df = pd.read_csv(path, sep=";", decimal=dec, encoding=enc, engine="python")
                    return self._normalize_columns(df)
                except Exception:
                    pass
        # fallback: tudo string
        df = pd.read_csv(path, sep=";", encoding="latin1", engine="python", dtype=str)
        return self._normalize_columns(df)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        norm_map = {c: self._norm(c) for c in df.columns}

        # candidatos: Iteration/Evaluation e Global Best/Fitness
        iter_cols = [c for c, n in norm_map.items()
                     if ("iteration" in n) or ("evaluation" in n) or ("avaliacao" in n) or ("iteracao" in n)]
        fit_cols  = [c for c, n in norm_map.items()
                     if ("globalbest" in n) or ("gbest" in n) or (n == "best") or ("fitness" in n)]

        def coalesce_numeric(cols: List[str]) -> pd.Series:
            if not cols:
                return pd.Series(dtype=float)
            block = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols})
            s = block.bfill(axis=1).iloc[:, 0]
            return s

        it = coalesce_numeric(iter_cols)
        ft = coalesce_numeric(fit_cols)

        # fallback nomes comuns
        if it.empty and "Iteration" in df.columns:
            it = pd.to_numeric(df["Iteration"], errors="coerce")
        if ft.empty:
            if "Global Best" in df.columns:
                ft = pd.to_numeric(df["Global Best"], errors="coerce")
            elif "Fitness" in df.columns:
                ft = pd.to_numeric(df["Fitness"], errors="coerce")

        df["Iteration"] = it
        df["Fitness"]   = ft
        df = df.dropna(subset=["Iteration", "Fitness"])
        df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")
        df["Fitness"]   = pd.to_numeric(df["Fitness"], errors="coerce")
        return df


# =========================
# Utilidades de ruído
# =========================
class NoiseUtils:
    # ordem desejada
    ORDER = ("N0", "N5", "N15")

    @staticmethod
    def detect_noise_token(path: Path) -> str:
        """Detecta N0/N5/N15:
           1) pelo nome da pasta imediata; 2) pelo nome do arquivo (0noise, 0.05noise, 0.15noise, 5noise, 15noise)."""
        parent = path.parent.name.upper()
        if parent in {"N0", "N5", "N15"}:
            return parent

        name = path.stem.lower()
        # padrões comuns em nomes: '0noise', '0.05noise', '0,05noise', '5noise', '15noise'
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*noise", name)
        if m:
            val = m.group(1).replace(",", ".")
            try:
                f = float(val)
                if f == 0:
                    return "N0"
                # 0.05 ou 5 → N5
                if np.isclose(f, 0.05) or np.isclose(f, 5.0):
                    return "N5"
                # 0.15 ou 15 → N15
                if np.isclose(f, 0.15) or np.isclose(f, 15.0):
                    return "N15"
            except Exception:
                pass
        return "N0"  # default

    @staticmethod
    def to_pct_label(token: str) -> str:
        return {"N0": "0%", "N5": "5%", "N15": "15%"}.get(token, token)

    @staticmethod
    def sort_key(token: str):
        try:
            return NoiseUtils.ORDER.index(token)
        except ValueError:
            return 999


# =========================
# Estatísticas por avaliação
# =========================
def compute_stats(dfs: List[pd.DataFrame]) -> Dict[str, pd.Series]:
    """Média e dispersão por avaliação (Iteration == Evaluations)."""
    big = pd.concat(dfs, ignore_index=True)
    g = big.groupby("Iteration")["Fitness"]

    mean = g.mean()
    std  = g.std().fillna(0.0)
    cnt  = g.count()

    # IC 95% (t-Student se disponível, senão z≈1.96)
    try:
        from scipy import stats as _stats
        tcrit = cnt.apply(lambda n: _stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
    except Exception:
        tcrit = cnt.apply(lambda n: 1.96 if n > 1 else np.nan)

    margin   = tcrit * (std / np.sqrt(cnt.replace(0, np.nan)))
    ci_lo    = mean - margin
    ci_hi    = mean + margin

    return {
        "mean": mean,
        "std": std,
        "min": g.min(),
        "max": g.max(),
        "q1": g.quantile(0.25),
        "q3": g.quantile(0.75),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }


# =========================
# Plotador
# =========================
class BOPlotter:
    def __init__(self, out_dir: Path, dispersion: str = "std"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dispersion = dispersion.lower()

    def plot(self, stats_by_noise: Dict[str, Dict[str, pd.Series]],
             title_prefix: str = "Convergence Line - Average Fitness - Bayesian Optimization") -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))

        legend_lines, legend_labels = [], []
        for noise in sorted(stats_by_noise.keys(), key=NoiseUtils.sort_key):
            s = stats_by_noise[noise]
            (mean_line,) = ax.plot(s["mean"].index, s["mean"].values, linewidth=2.0,
                                   label=NoiseUtils.to_pct_label(noise))
            color = mean_line.get_color()

            if self.dispersion == "std":
                lo, hi = s["mean"] - s["std"], s["mean"] + s["std"]
            elif self.dispersion == "iqr":
                lo, hi = s["q1"], s["q3"]
            elif self.dispersion == "minmax":
                lo, hi = s["min"], s["max"]
            elif self.dispersion == "ci":
                lo, hi = s["ci_lower"], s["ci_upper"]
            else:
                lo, hi = s["q1"], s["q3"]

            # ax.fill_between(s["mean"].index, lo.values, hi.values, alpha=0.20, color=color)
            # ax.plot(s["max"].index, s["max"].values, linestyle="--", linewidth=1.2, color=color, alpha=0.9)
            # ax.plot(s["min"].index, s["min"].values, linestyle="-.",  linewidth=1.2, color=color, alpha=0.9)

            legend_lines.append(mean_line)
            legend_labels.append(NoiseUtils.to_pct_label(noise))

        # ax.set_title(title_prefix)
        ax.set_xlabel("Evaluations")  # eixo X em inglês, como solicitado
        ax.set_ylabel("Fitness")
        ax.grid(True, axis="both", linestyle="--", alpha=0.4)

        if legend_lines:
            ax.legend(legend_lines, legend_labels, title="Noise",
                      loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, frameon=True)

        out_path = self.out_dir / "BO_convergence_N0_N5_N15_noextremes.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path


# =========================
# Pipeline
# =========================
@dataclass
class PipelineConfig:
    in_dir: Path          # pasta que contém N0, N5, N15 (cada uma com 4 CSVs)
    out_dir: Path
    pattern: str = "*.csv"
    dispersion: str = "std"   # "std" | "iqr" | "minmax" | "ci"


class BOPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = CSVLoader()
        self.plotter = BOPlotter(cfg.out_dir, dispersion=cfg.dispersion)

    def run(self) -> Path:
        stats_by_noise: Dict[str, Dict[str, pd.Series]] = {}

        # 1) varre subpastas de ruído e CSVs
        for p in sorted(self.cfg.in_dir.rglob(self.cfg.pattern)):
            if not p.is_file():
                continue
            noise = NoiseUtils.detect_noise_token(p)
            df = self.loader.read(p)
            if df.empty:
                continue
            # 2) garante Series numéricas e descarta NaN
            df = df.dropna(subset=["Iteration", "Fitness"])
            if df.empty:
                continue
            # 3) acumula por ruído (cada arquivo é uma rodada)
            stats_by_noise.setdefault(noise, []).append(df)

        # 4) calcula estatísticas por avaliação (média entre rodadas)
        stats_by_noise = {nz: compute_stats(dfs) for nz, dfs in stats_by_noise.items() if dfs}

        # 5) plota 3 linhas (N0, N5, N15)
        return self.plotter.plot(stats_by_noise)


# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    IN_DIR  = Path(r"D:\Thiago Artur\OneDrive\Documentos\2025.1\Cilamce\Para plotagem")   # <- ajuste aqui (pasta que contém N0, N5, N15)
    OUT_DIR = Path(r'D:\Thiago Artur\OneDrive\Documentos\2025.1\Cilamce\Para plotagem\graph')

    cfg = PipelineConfig(in_dir=IN_DIR, out_dir=OUT_DIR, pattern="*.csv", dispersion="std")
    path_png = BOPipeline(cfg).run()
    print("Figura salva em:", path_png)