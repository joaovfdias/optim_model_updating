from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuração global de estilo
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
# margens fixas, deixando espaço à direita para legenda
plt.rcParams['figure.subplot.left']   = 0.10
plt.rcParams['figure.subplot.right']  = 0.80
plt.rcParams['figure.subplot.top']    = 0.90
plt.rcParams['figure.subplot.bottom'] = 0.10


# =========================
# Utilidades
# =========================
@dataclass(frozen=True)
class FileTokens:
    model: str          # 'viga' ou 'laje'
    analysis: str       # 'freq' ou 'freq_mac'
    noise_token: str    # 'N0', 'N0.05', 'N0.15', 'N5', etc.

class FileNameParser:
    MODEL_OPTIONS = ("viga", "laje")

    @staticmethod
    def parse(path: Path) -> FileTokens | None:
        name = path.stem.lower()

        model = next((m for m in FileNameParser.MODEL_OPTIONS if m in name), None)
        if not model:
            return None

        # 'freq_mac' tem prioridade sobre 'freq'
        if "freq_mac" in name or "mac" in name and "freq" in name:
            analysis = "freq_mac"
        elif "freq" in name:
            analysis = "freq"
        else:
            return None

        # ruído como N<numero> (aceita N0, N5, N0.05, N0.15, N0.0 etc.)
        m = re.search(r"n\d+(?:\.\d+)?", name)
        noise_token = (m.group(0).upper() if m else "N0")

        return FileTokens(model=model, analysis=analysis, noise_token=noise_token)


class CSVLoader:
    """Leitor robusto para CSV com sep=';' e decimal=','."""
    def read(self, path: Path) -> pd.DataFrame:
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(path, sep=";", decimal=",", encoding=enc, engine="python")
                return self._normalize_columns(df)
            except Exception:
                pass
        # fallback puro (dtype=str)
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(path, sep=";", encoding=enc, engine="python", dtype=str)
                return self._normalize_columns(df)
            except Exception:
                pass
        raise RuntimeError(f"Falha ao ler {path}")

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # strip e tentativa de detectar colunas alvo
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if "Iteration" not in df.columns or "Fitness" not in df.columns:
            norm = {c: re.sub(r"\s+", "", c.strip().lower()) for c in df.columns}
            col_iter = next((c for c, v in norm.items() if "iteration" in v or "iteracao" in v or "iterac" in v), None)
            col_fit  = next((c for c, v in norm.items() if "fitness" in v), None)
            if col_iter and col_fit:
                df = df.rename(columns={col_iter: "Iteration", col_fit: "Fitness"})
        return df


class NoiseUtils:
    """Conversões e ordenação de níveis de ruído."""
    # ordem desejada: 0% < 5% < 15%
    DESIRED_ORDER = ("N0", "N0.0", "N0.05", "N5", "N0.15")

    @staticmethod
    def token_sort_key(token: str) -> Tuple[int, float]:
        """Chave de ordenação: primeiro pela ordem desejada, depois valor numérico."""
        idx = NoiseUtils.DESIRED_ORDER.index(token) if token in NoiseUtils.DESIRED_ORDER else 999
        # valor numérico: N0.05 -> 0.05; N5 -> 5
        try:
            val = float(token[1:])
        except Exception:
            val = 1e9
        return (idx, val)

    @staticmethod
    def to_percent_label(token: str) -> str:
        try:
            val = float(token[1:])
            pct = int(round(val * 100)) if val <= 1.0 else int(round(val))
            return f"{pct}%"
        except Exception:
            return token


# =========================
# Núcleo de agregação
# =========================
@dataclass
class GroupStats:
    per_noise: Dict[str, Dict[str, pd.Series]]


class ConvergenceAggregator:
    def __init__(self, loader: CSVLoader):
        self.loader = loader

    def load_grouped(self, files: List[Path]) -> Dict[Tuple[str, str], Dict[str, List[pd.DataFrame]]]:
        grouped: Dict[Tuple[str, str], Dict[str, List[pd.DataFrame]]] = {}
        for p in files:
            tokens = FileNameParser.parse(p)
            if not tokens:
                continue
            df = self.loader.read(p)
            if "Iteration" not in df.columns or "Fitness" not in df.columns:
                continue
            df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")
            df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
            df = df.dropna(subset=["Iteration", "Fitness"])
            if df.empty:
                continue
            key = (tokens.model, tokens.analysis)
            grouped.setdefault(key, {}).setdefault(tokens.noise_token, []).append(df)
        return grouped

    @staticmethod
    def compute_stats(dfs: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        big = pd.concat(dfs, ignore_index=True)
        g = big.groupby("Iteration")["Fitness"]

        mean = g.mean()
        std = g.std().fillna(0.0)
        cnt = g.count()

        # IC 95% da média (t-Student; se scipy indisponível, usa z≈1.96)
        try:
            from scipy import stats as _stats
            tcrit = cnt.apply(lambda n: _stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
        except Exception:
            tcrit = cnt.apply(lambda n: 1.96 if n > 1 else np.nan)

        margin = tcrit * (std / np.sqrt(cnt.replace(0, np.nan)))
        ci_lower = mean - margin
        ci_upper = mean + margin

        return {
            "mean": mean,
            "std": std,
            "min": g.min(),
            "max": g.max(),
            "q1": g.quantile(0.25),
            "q3": g.quantile(0.75),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def aggregate(self, grouped: Dict[Tuple[str, str], Dict[str, List[pd.DataFrame]]]) -> Dict[
        Tuple[str, str], GroupStats]:
        out: Dict[Tuple[str, str], GroupStats] = {}
        for key, noise_map in grouped.items():
            per_noise: Dict[str, Dict[str, pd.Series]] = {}
            for noise, dfs in noise_map.items():
                per_noise[noise] = self.compute_stats(dfs)
            out[key] = GroupStats(per_noise=per_noise)
        return out


# =========================
# Plotter
# =========================
class ConvergencePlotter:
    def __init__(self, out_dir: Path, dispersion: str = "std"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dispersion = dispersion.lower()
        self.y_ranges: Dict[str, Tuple[float, float]] = {}

    def set_y_ranges(self, y_ranges: Dict[str, Tuple[float, float]]):
        """Define limites fixos por tipo de model ('beam', 'bridge')."""
        self.y_ranges = y_ranges

    @staticmethod
    def _analysis_label(analysis: str) -> str:
        return "Freq MAC" if analysis.lower().startswith("freq_mac") else "Freq"

    def plot_group(self, key: Tuple[str, str], stats: GroupStats) -> Path:
        model, analysis = key
        fig, ax = plt.subplots(figsize=(12, 6))

        noises = sorted(stats.per_noise.keys(), key=NoiseUtils.token_sort_key)

        legend_lines, legend_labels = [], []
        for noise in noises:
            s = stats.per_noise[noise]
            (mean_line,) = ax.plot(s["mean"].index, s["mean"].values, linewidth=2.0,
                                   label=NoiseUtils.to_percent_label(noise))
            color = mean_line.get_color()

            if self.dispersion == "std":
                lower = s["mean"] - s["std"]
                upper = s["mean"] + s["std"]
            elif self.dispersion == "iqr":
                lower = s["q1"]
                upper = s["q3"]
            elif self.dispersion == "minmax":
                lower = s["min"]
                upper = s["max"]
            elif self.dispersion == "ci":
                lower = s["ci_lower"]
                upper = s["ci_upper"]
            else:
                raise ValueError("dispersion must be 'std', 'iqr', 'minmax' or 'ci'")

            ax.fill_between(s["mean"].index, lower.values, upper.values, alpha=0.20, color=color)

            # >>> NOVO: máximos e mínimos globais por iteração em tracejado
            ax.plot(s["max"].index, s["max"].values, linestyle="--", linewidth=1.2, color=color, alpha=0.9)
            ax.plot(s["min"].index, s["min"].values, linestyle="-.", linewidth=1.2, color=color, alpha=0.9)

            legend_lines.append(mean_line)
            legend_labels.append(NoiseUtils.to_percent_label(noise))

        m_model = 'Bridge' if model.lower().startswith("laje") else 'Beam'
        title = f"Convergence Line - Average fitness - {m_model} {self._analysis_label(analysis)}"
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.grid(True, axis="both", linestyle="--", alpha=0.4)

        # aplica range fixo
        m_key = 'bridge' if model.lower().startswith("laje") else 'beam'
        if m_key in self.y_ranges:
            ax.set_ylim(self.y_ranges[m_key])

        if legend_lines:
            ax.legend(legend_lines, legend_labels, title="Noise",
                      loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, frameon=True)

        a_name = 'FreqMAC' if analysis.lower().startswith("freq_mac") else 'Freq'
        m_name = 'Bridge' if model.lower().startswith("laje") else 'Beam'
        fname = f"{m_name}_{a_name}_convergence.png"
        out_path = self.out_dir / fname
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path
# =========================
# Pipeline
# =========================
@dataclass
class PipelineConfig:
    in_dir: Path
    out_dir: Path
    pattern: str = "*.csv"
    dispersion: str = "std"   # "std" | "iqr" | "minmax" | "ci"


class ConvergencePipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = CSVLoader()
        self.aggregator = ConvergenceAggregator(self.loader)
        self.plotter = ConvergencePlotter(cfg.out_dir, dispersion=cfg.dispersion)

    def run(self) -> List[Path]:
        files = sorted(self.cfg.in_dir.glob(self.cfg.pattern))
        grouped = self.aggregator.load_grouped(files)
        groups_stats = self.aggregator.aggregate(grouped)

        # 1️⃣ calcular ranges fixos por model
        y_ranges: Dict[str, Tuple[float, float]] = {}
        for (model, _analysis), stats in groups_stats.items():
            m_key = 'bridge' if model.lower().startswith("laje") else 'beam'

            # pegar min e max considerando a faixa de dispersão escolhida
            vals = []
            for noise, s in stats.per_noise.items():
                if self.cfg.dispersion == "std":
                    vals.extend((s["mean"] - s["std"]).values)
                    vals.extend((s["mean"] + s["std"]).values)
                elif self.cfg.dispersion == "iqr":
                    vals.extend(s["q1"].values)
                    vals.extend(s["q3"].values)
                elif self.cfg.dispersion == "minmax":
                    vals.extend(s["min"].values)
                    vals.extend(s["max"].values)
                elif self.cfg.dispersion == "ci":
                    vals.extend(s["ci_lower"].values)
                    vals.extend(s["ci_upper"].values)
                else:
                    vals.extend(s["mean"].values)

            min_val, max_val = np.nanmin(vals), np.nanmax(vals)

            if m_key not in y_ranges:
                y_ranges[m_key] = (min_val, max_val)
            else:
                old_min, old_max = y_ranges[m_key]
                y_ranges[m_key] = (min(old_min, min_val), max(old_max, max_val))

        # 2️⃣ passar ranges para o plotter
        self.plotter.set_y_ranges(y_ranges)

        # 3️⃣ plotar normalmente
        outputs: List[Path] = []
        for key, stats in groups_stats.items():
            out_path = self.plotter.plot_group(key, stats)
            outputs.append(out_path)
        return outputs


# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    cfg = PipelineConfig(
        in_dir=Path(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data'),
        out_dir=Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Out_Plots"),
        pattern="*.csv",
        dispersion="std"  # "std" | "iqr" | "minmax" | "ci"
    )
    pipeline = ConvergencePipeline(cfg)
    pipeline.run()
