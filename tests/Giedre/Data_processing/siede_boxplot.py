# -*- coding: utf-8 -*-
"""
POO: Uma figura por Model, ordem default dos erros, títulos embaixo com (a),(b),(c)...
Legenda de cores (Analysis) no último slot: última linha × última coluna.

- Sem fliers: whis=[0,100], showfliers=False
- Y label só no 1º subplot de cada linha
- k3/k4 apenas se Model == 'bridge'
- Reserva sempre a última célula da grade para a legenda. Se necessário, aumenta nrows.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =========================
# Config
# =========================
@dataclass
class Config:
    csv_path: Path
    out_dir: Path
    show: bool = True
    error_cols_manual: Optional[List[str]] = None  # opcional: força ordem manual
    font_family: str = "Times New Roman"
    font_size: int = 12
    dpi: int = 200

    # ---- layout dos subplots ----
    ncols: int = 3                 # nº de colunas na grade
    group_span: float = 0.7        # largura por grupo (para múltiplas analyses)
    fig_row_height: float = 4.4    # altura de cada linha
    row_spacing: float = 0.12      # hspace entre linhas
    col_spacing: float = 0.26      # wspace entre colunas

    # ---- margens e folga extra ----
    extra_margin_height: float = 0.8    # folga adicional na altura total (inches)
    margins: Dict[str, float] = None    # frações [0..1]: left,right,top,bottom

    def __post_init__(self):
        if self.margins is None:
            self.margins = dict(left=0.08, right=0.985, top=0.96, bottom=0.14)

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Data
# =========================
class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def read_csv_semicolon(self) -> pd.DataFrame:
        path = self.config.csv_path
        df = None
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(path, sep=";", encoding=enc, engine="python", dtype=str)
                break
            except Exception:
                continue
        if df is None:
            raise RuntimeError("Falha ao ler o CSV com sep=';'.")
        df.columns = [c.strip() for c in df.columns]
        for key in ("Model", "Analysis", "Noise"):
            if key in df.columns:
                df[key] = df[key].astype(str).str.strip()
        return df


class Cleaner:
    @staticmethod
    def clean_number_series(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        s = s.str.replace("\xa0", " ", regex=False).str.strip()
        s = s.str.replace("%", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        s = s.apply(lambda x: re.sub(r"[^0-9eE\.\+\-]", "", x))
        return pd.to_numeric(s, errors="coerce")


class ErrorDetector:
    def __init__(self, error_cols_manual: Optional[List[str]] = None):
        self.error_cols_manual = error_cols_manual

    def detect_error_cols(self, df: pd.DataFrame) -> List[str]:
        if self.error_cols_manual:
            cols = [c for c in self.error_cols_manual if c in df.columns]
        else:
            # mantém a ORDEM natural do CSV
            cols = [c for c in df.columns if "error" in c.lower()]
        if not cols:
            raise ValueError("Nenhuma coluna de erro encontrada.")
        return cols


# =========================
# Levels
# =========================
class LevelsPreparer:
    def __init__(self, cleaner: Cleaner):
        self.cleaner = cleaner

    def prepare_levels_once(self, df_model: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List, List[str], List[str]]:
        noise_num = self.cleaner.clean_number_series(df_model["Noise"])
        use_noise_num = noise_num.notna().sum() >= df_model["Noise"].notna().sum() * 0.5
        df_m = df_model.copy()
        if use_noise_num:
            df_m["Noise_num"] = noise_num
            noise_levels = sorted(df_m["Noise_num"].dropna().unique())
            xticklabels = [str(n) for n in noise_levels]
        else:
            noise_levels = sorted(df_m["Noise"].dropna().astype(str).unique())
            xticklabels = [str(n) for n in noise_levels]
        analyses = sorted(df_m["Analysis"].dropna().astype(str).unique())
        return df_m, use_noise_num, noise_levels, xticklabels, analyses


# =========================
# Plotter
# =========================
class ModelFigurePlotter:
    def __init__(self, config: Config, cleaner: Cleaner, levels_prep: LevelsPreparer):
        self.config = config
        self.cleaner = cleaner
        self.levels_prep = levels_prep
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.size'] = self.config.font_size

    def _analysis_color_map(self, analyses: List[str]) -> Dict[str, str]:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if not prop_cycle:
            prop_cycle = [f"C{i}" for i in range(10)]
        return {a: prop_cycle[i % len(prop_cycle)] for i, a in enumerate(analyses)}

    def _compute_box_width(self, analyses: List[str]) -> float:
        n_a = max(1, len(analyses))
        step = self.config.group_span / n_a
        return step * 0.9

    @staticmethod
    def _alpha_tag(idx: int) -> str:
        # (a)...(z),(aa)... etc.
        letters = []
        i = idx
        while True:
            letters.append(chr(ord('a') + (i % 26)))
            i //= 26
            if i == 0:
                break
            i -= 1
        return "(" + "".join(reversed(letters)) + ")"

    def _plot_one(
        self,
        ax: plt.Axes,
        df_m: pd.DataFrame,
        err_col: str,
        prep: Tuple[pd.DataFrame, bool, List, List[str], List[str]],
        analysis_colors: Dict[str, str],
        show_ylabel: bool,
        box_width: float,
        caption: str,
    ) -> bool:
        df_m, use_noise_num, noise_levels, xticklabels, analyses = prep
        base_positions = np.arange(1, len(noise_levels) + 1)

        n_a = max(1, len(analyses))
        span = self.config.group_span
        if n_a == 1:
            offsets = np.array([0.0])
        else:
            step = span / n_a
            offsets = np.linspace(-span/2 + step/2, span/2 - step/2, n_a)

        any_data = False
        for i, a in enumerate(analyses):
            data = []
            for nl in noise_levels:
                mask = ((df_m["Analysis"] == a) &
                        ((df_m.get("Noise_num", np.nan) == nl) if use_noise_num else (df_m["Noise"] == nl)))
                vals = pd.to_numeric(df_m.loc[mask, err_col], errors="coerce").dropna().values
                data.append(vals)
                if len(vals) > 0:
                    any_data = True

            positions = base_positions + offsets[i]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                whis=[0, 100],
            )
            color = analysis_colors.get(a, None)
            for patch in bp["boxes"]:
                if color:
                    patch.set_facecolor(color)
            for k in ["whiskers", "caps", "medians"]:
                for item in bp[k]:
                    try:
                        if color:
                            item.set_color(color)
                    except Exception:
                        pass

        ax.set_xlabel("Noise")
        ax.set_ylabel("optimal value / reference value" if show_ylabel else "")
        ax.set_xticks(base_positions)
        ax.set_xticklabels(xticklabels)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # título embaixo (legenda de subplot)
        ax.text(0.5, -0.18, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=self.config.font_size)

        return any_data

    def plot_model_figure(self, df_model: pd.DataFrame, error_cols_ordered: List[str], out_path: Path) -> bool:
        title_model = str(df_model["Model"].iloc[0]) if not df_model.empty else ""
        prep = self.levels_prep.prepare_levels_once(df_model)
        _, _, _, _, analyses = prep
        analysis_colors = self._analysis_color_map(analyses)

        # k3/k4 só em bridge
        is_bridge = "bridge" in (title_model or "").lower()
        if not is_bridge:
            error_cols_ordered = [c for c in error_cols_ordered if "k3" not in c.lower() and "k4" not in c.lower()]
        nplots = len(error_cols_ordered)
        if nplots == 0:
            return False

        ncols = max(1, self.config.ncols)

        # >>> Reserve uma célula para a legenda (última). Calcular nrows mínimo tal que nplots <= nrows*ncols - 1
        nrows = int(np.ceil((nplots + 1) / ncols))

        fig_width = 10.8
        fig_height = self.config.fig_row_height * nrows + self.config.extra_margin_height
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.config.dpi)
        gs = GridSpec(nrows, ncols, figure=fig,
                      wspace=self.config.col_spacing, hspace=self.config.row_spacing)

        auto_box_width = self._compute_box_width(analyses)
        any_plot = False

        # Posicionar gráficos; deixar a última célula (nrows-1, ncols-1) para a legenda
        last_cell_index = nrows * ncols - 1

        for i, err_col in enumerate(error_cols_ordered):
            # pular a célula de legenda se for alcançada
            cell = i if i < last_cell_index else i + 1
            r = cell // ncols
            c = cell % ncols
            ax = fig.add_subplot(gs[r, c])

            caption = f"{self._alpha_tag(i)} {err_col}"
            show_ylabel = (c == 0)
            ok = self._plot_one(ax, df_model, err_col, prep, analysis_colors, show_ylabel, auto_box_width, caption)

            tm = title_model.lower()
            if 'beam' in tm:
                ax.set_ylim(0, 2.5)
            if 'bridge' in tm:
                ax.set_ylim(0, 3.5)

            any_plot = any_plot or ok

        # Apaga células vazias (exceto a reservada para legenda)
        total_cells = nrows * ncols
        for j in range(nplots, total_cells - 1):
            ax = fig.add_subplot(gs[j // ncols, j % ncols])
            ax.axis("off")

        # >>> Último eixo (inferior direito) recebe a LEGENDA DE CORES
        legend_ax = fig.add_subplot(gs[(total_cells - 1) // ncols, (total_cells - 1) % ncols])
        legend_ax.axis("off")
        if analyses:
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(facecolor=analysis_colors[a], edgecolor="black", label=a) for a in analyses]
            legend_ax.legend(handles=handles, title="Analysis", loc="center", frameon=True)

        # Supertítulo
        #fig.suptitle(title_model, y=0.985)

        # margens finais (sem tight_layout para evitar warnings)
        fig.subplots_adjust(
            left=self.config.margins["left"],
            right=self.config.margins["right"],
            top=self.config.margins["top"],
            bottom=self.config.margins["bottom"],
            wspace=self.config.col_spacing,
            hspace=self.config.row_spacing,
        )

        if any_plot:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
            if self.config.show:
                plt.show()
        plt.close(fig)
        return any_plot


# =========================
# App
# =========================
class BoxplotApp:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.cleaner = Cleaner()
               # detecção mantém a ORDEM natural do CSV (ou usa manual)
        self.detector = ErrorDetector(config.error_cols_manual)
        self.levels = LevelsPreparer(self.cleaner)
        self.plotter = ModelFigurePlotter(config, self.cleaner, self.levels)

    def _prepare_dataframe(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        error_cols = self.detector.detect_error_cols(df_raw)
        df = df_raw.copy()
        for c in error_cols:
            df[c] = self.cleaner.clean_number_series(df[c])
        return df, error_cols

    def run(self) -> None:
        self.config.ensure_dirs()
        df_raw = self.loader.read_csv_semicolon()
        df, error_cols = self._prepare_dataframe(df_raw)

        models = sorted(df["Model"].dropna().unique(), key=str) if "Model" in df.columns else []
        total_figs = 0

        for model in models:
            df_m = df[df["Model"] == model].copy()
            if df_m.empty or df_m["Analysis"].isna().all() or df_m["Noise"].isna().all():
                continue

            out_dir_model = self.config.out_dir / str(model)
            out_file = out_dir_model / f"{str(model)}__ALL_ERRORS_DEFAULT_ORDER.png"
            ok = self.plotter.plot_model_figure(df_m, error_cols, out_file)
            if ok:
                total_figs += 1
            else:
                print(f"[AVISO] Sem dados válidos para: Model={model}")

        print(f"Concluído. Figuras geradas: {total_figs}. Pasta: {self.config.out_dir.resolve()}")


# =========================
# Execução
# =========================
if __name__ == "__main__":
    cfg = Config(
        csv_path=Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Final_data\Final_Data.csv"),
        out_dir=Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Out_Plots\Boxplot_Default_Order"),
        show=True,
        # error_cols_manual=["E Error","ν Error","r Error","k1 Error","k2 Error","k3 Error","k4 Error"],
        dpi=200,
        ncols=3,
        group_span=0.7,
        fig_row_height=4.4,
        row_spacing=0.35,
        col_spacing=0.26,
        extra_margin_height=0.8,
        margins=dict(left=0.08, right=0.985, top=0.96, bottom=0.10),
    )
    BoxplotApp(cfg).run()
