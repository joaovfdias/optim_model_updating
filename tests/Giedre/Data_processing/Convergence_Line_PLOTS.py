# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

import unicodedata
import pandas as pd
import matplotlib.pyplot as plt

# Se você usa seu Group, mantenha o import:
from Group import Group


# =========================
# Utilidades de nomes/colunas
# =========================
class NameUtils:
    @staticmethod
    def normalize_name(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s.strip().lower()

    @staticmethod
    def find_column(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
        """
        Retorna o nome original da primeira coluna cujo 'nome normalizado'
        contenha qualquer uma das strings em targets.
        """
        norm_map = {col: NameUtils.normalize_name(col) for col in df.columns}
        for col, norm in norm_map.items():
            if any(t in norm for t in targets):
                return col
        return None


# =========================
# Leitura de CSV
# =========================
class CSVReader:
    def __init__(self, sep: str = ";", decimal: str = ","):
        self.sep = sep
        self.decimal = decimal

    def read(self, path: Path) -> pd.DataFrame:
        # 1) tenta leitura já com decimal=','
        for enc in ("utf-8", "latin1"):
            try:
                return pd.read_csv(path, sep=self.sep, decimal=self.decimal,
                                   encoding=enc, engine="python")
            except Exception:
                pass
        # 2) fallback: texto puro
        for enc in ("utf-8", "latin1"):
            try:
                return pd.read_csv(path, sep=self.sep, encoding=enc,
                                   engine="python", dtype=str)
            except Exception:
                pass
        raise RuntimeError(f"Falha ao ler {path} como CSV com sep='{self.sep}'.")

    @staticmethod
    def ensure_numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")


# =========================
# Processamento por arquivo (trial)
# =========================
@dataclass
class TrialResult:
    mean_by_iteration: pd.Series          # média por iteração (index=Iteration)
    slim_df: pd.DataFrame                 # colunas: ['Iteration','Fitness']


class TrialProcessor:
    def __init__(self, reader: CSVReader,
                 iteration_aliases: Optional[List[str]] = None,
                 fitness_aliases: Optional[List[str]] = None,
                 force_int_iteration: bool = True):
        self.reader = reader
        self.iteration_aliases = iteration_aliases or ["iteration", "iteracao", "iterac"]
        self.fitness_aliases = fitness_aliases or ["fitness"]
        self.force_int_iteration = force_int_iteration

    def process(self, csv_path: Path) -> TrialResult:
        df = self.reader.read(csv_path)
        df.columns = [c.strip() for c in df.columns]

        col_iter = NameUtils.find_column(df, self.iteration_aliases)
        col_fit = NameUtils.find_column(df, self.fitness_aliases)

        if col_iter is None or col_fit is None:
            raise ValueError(
                f"No arquivo {csv_path.name} não encontrei colunas 'Iteration' e/ou 'Fitness'. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        # Numéricos
        df[col_iter] = self.reader.ensure_numeric(df[col_iter])
        df[col_fit] = self.reader.ensure_numeric(df[col_fit])

        # Remove inválidos
        df = df.dropna(subset=[col_iter, col_fit])

        # (Opcional) força inteiro usando dtype pandas 'Int64'
        if self.force_int_iteration:
            try:
                df[col_iter] = df[col_iter].round().astype("Int64")
            except Exception:
                # Se não rolar, segue como numérico float mesmo
                pass

        mean_by_it = df.groupby(col_iter, dropna=True)[col_fit].mean().sort_index()
        slim = df[[col_iter, col_fit]].rename(columns={col_iter: "Iteration", col_fit: "Fitness"})

        return TrialResult(mean_by_iteration=mean_by_it, slim_df=slim)


# =========================
# Agregação global
# =========================
@dataclass
class ConvergenceStats:
    per_file_means: Dict[str, pd.Series]
    global_mean: pd.Series
    global_min: pd.Series
    global_max: pd.Series


class ConvergenceAggregator:
    def __init__(self, processor: TrialProcessor):
        self.processor = processor

    def compute(self, files: Iterable[Path]) -> ConvergenceStats:
        per_file_means: Dict[str, pd.Series] = {}
        slims: List[pd.DataFrame] = []

        for i, f in enumerate(files, start=1):
            try:
                res = self.processor.process(f)
            except Exception as e:
                print(f"[AVISO] Pulando {Path(f).name}: {e}")
                continue
            per_file_means[Path(f).stem] = res.mean_by_iteration
            slims.append(res.slim_df)

        if not per_file_means:
            raise RuntimeError("Nenhum arquivo válido com Iteration/Fitness foi processado.")

        all_df = pd.concat(slims, ignore_index=True)

        global_mean = all_df.groupby("Iteration")["Fitness"].mean().sort_index()
        global_min = all_df.groupby("Iteration")["Fitness"].min().sort_index()
        global_max = all_df.groupby("Iteration")["Fitness"].max().sort_index()

        return ConvergenceStats(
            per_file_means=per_file_means,
            global_mean=global_mean,
            global_min=global_min,
            global_max=global_max,
        )


# =========================
# Plot
# =========================
@dataclass
class PlotLabels:
    model: str
    analysis: str
    noise: str


class ConvergencePlotter:
    def __init__(self, labels: PlotLabels):
        self.labels = labels

    def plot_and_save(self, stats: ConvergenceStats, output_path: Path) -> None:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12

        plt.rcParams['figure.subplot.left'] = 0.1
        plt.rcParams['figure.subplot.right'] = 0.8
        plt.rcParams['figure.subplot.top'] = 0.9
        plt.rcParams['figure.subplot.bottom'] = 0.1

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 1) Linhas por arquivo
        for idx, (name, s) in enumerate(stats.per_file_means.items(), start=1):
            ax.plot(s.index, s.values, linewidth=1.5, alpha=0.9, label=f"Average — Trial {idx}")

        # 2) Média global
        ax.plot(stats.global_mean.index, stats.global_mean.values, linewidth=2.5, label="Global Average")

        # 3) Envelopes
        ax.plot(stats.global_max.index, stats.global_max.values, linestyle="--", linewidth=1.2, label="Global Maximum")
        ax.plot(stats.global_min.index, stats.global_min.values, linestyle="-.", linewidth=1.2, label="Global Minimum")

        ax.set_title(
            f"{self.labels.model} {self.labels.analysis} {self.labels.noise} / "
            f"Fitness per Generation — average per trial, global average, global extremes"
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")

        # ==== FIXAR RANGE DO EIXO Y DEPENDENDO DO MODELO ====
        if self.labels.model.lower() == "bridge" or self.labels.model.lower() == "laje":
            ax.set_ylim(0, 80)  # exemplo para Bridge
        elif self.labels.model.lower() == "beam" or self.labels.model.lower() == "viga":
            ax.set_ylim(0, 180)  # exemplo para Beam

        ax.grid(True, axis="both", linestyle="--", alpha=0.4)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.show()

# =========================
# Pipeline de ponta a ponta
# =========================
@dataclass
class PipelineConfig:
    files: Optional[List[Path]]                 # lista explícita (se já tiver os Paths)
    output_figure: Path
    labels: PlotLabels
    # Caso precise buscar arquivos por padrão:
    folder: Optional[Path] = None
    pattern: str = "*.csv"


class ConvergencePipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.reader = CSVReader(sep=";", decimal=",")
        self.processor = TrialProcessor(self.reader, force_int_iteration=True)
        self.aggregator = ConvergenceAggregator(self.processor)
        self.plotter = ConvergencePlotter(cfg.labels)

    def _resolve_files(self) -> List[Path]:
        if self.cfg.files:
            return list(self.cfg.files)
        if self.cfg.folder:
            return sorted(self.cfg.folder.glob(self.cfg.pattern))
        return []

    def run(self) -> Path:
        files = self._resolve_files()
        if not files:
            raise FileNotFoundError("Nenhum CSV encontrado. Ajuste a lista de arquivos ou folder/pattern.")

        stats = self.aggregator.compute(files)
        self.plotter.plot_and_save(stats, self.cfg.output_figure)
        return self.cfg.output_figure


# =========================
# Exemplo de uso (ajuste aos seus grupos/paths)
# =========================
# def build_files_from_groups() -> List[Path]:
#     """
#     Mantém sua lógica baseada no Group, retornando a lista FILES.
#     Ajuste os índices conforme sua necessidade.
#     """
main_folder_path = r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data"
output_path_ = Path(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Out_Plots')
model = Group(main_folder_path, 'model').group()  # model[0] -> Bridge
for m in model:
    analysis = Group(main_folder_path, 'mac', m).group()  # analysis[?]
    for a in analysis:
        noise = Group(main_folder_path, 'noise', a).group()  # noise[2] = 0 Noise
        for n in noise:
            files = n  # Deve ser uma lista de Paths

            if __name__ == "__main__":
                # --- arquivos vindos do seu agrupamento (Group) ---
                # --- labels do título ---
                file = files[0]
                file_name = Path(file).stem
                name = file_name.lower()
                if 'laje' in name or 'bridge' in name:
                    model = 'Bridge'
                else:
                    model = 'Beam'
                analysis = 'Freq and Mac' if 'mac' in name else 'Freq'
                analysis2 = 'FreqMac' if 'mac' in name else 'Freq'
                if 'n15' in name or 'n0.15' in name:
                    noise = '15% noise'
                    noise2 = 'N15'
                elif 'n5' in name or 'n0.05' in name:
                    noise = '5% noise'
                    noise2 = 'N5'
                else:
                    noise = '0% noise'
                    noise2 = 'N0'

                labels = PlotLabels(model=model, analysis=analysis, noise=noise)

                # --- saída ---
                out_name = f'{model}_{analysis2}_{noise2}'
                output_fig = output_path_ / f'{out_name}.png'

                cfg = PipelineConfig(
                    files=files,  # ou use folder/pattern se preferir varrer uma pasta
                    folder=None,
                    pattern="*.csv",
                    output_figure=output_fig,
                    labels=labels,
                )

                pipeline = ConvergencePipeline(cfg)
                out = pipeline.run()
                print(f"Figura salva em: {out.resolve()}")






