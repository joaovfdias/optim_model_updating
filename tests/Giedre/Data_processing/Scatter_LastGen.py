# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Estilo global
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# =========================
# Layout (margens fixas)
# =========================
MARGINS = dict(left=0.12, right=0.78, top=0.90, bottom=0.12)

# Limites manuais opcionais por Model (se quiser forçar):
# Ex.: MANUAL_Y_LIMITS = {'Bridge': (0, 120), 'Beam': (10, 80)}
MANUAL_Y_LIMITS: Dict[str, Tuple[float, float]] = {}

# Legendas de Noise específicas solicitadas
NOISE_LABELS = {0.0: "0%", 0.05: "5%", 0.15: "15%"}

@dataclass(frozen=True)
class FileMeta:
    path: Path
    model: str       # "Bridge" ou "Beam"
    analysis: str    # "FreqMac" ou "Freq"
    noise: float     # ex.: 0.05


class NameRules:
    """Inferência de Model/Analysis/Noise a partir do nome do arquivo."""
    _noise_re = re.compile(r"N(?P<val>\d+(?:\.\d+)?)", flags=re.IGNORECASE)

    @staticmethod
    def get_model(stem: str) -> Optional[str]:
        s = stem.lower()
        if "laje" in s:
            return "Bridge"
        if "viga" in s:
            return "Beam"
        return None

    @staticmethod
    def get_analysis(stem: str) -> str:
        s = stem.lower()
        return "FreqMac" if "mac" in s else "Freq"

    @classmethod
    def get_noise(cls, stem: str) -> Optional[float]:
        m = cls._noise_re.search(stem)
        if not m:
            return None
        try:
            return float(m.group("val"))
        except ValueError:
            return None

    @classmethod
    def parse(cls, path: Path) -> Optional[FileMeta]:
        stem = path.stem
        model = cls.get_model(stem)
        if model is None:
            return None
        analysis = cls.get_analysis(stem)
        noise = cls.get_noise(stem)
        if noise is None:
            return None
        return FileMeta(path=path, model=model, analysis=analysis, noise=noise)


class GAGroupScatter:
    """
    - Agrupa arquivos por (Model, Analysis) via nome
    - Para cada grupo, plota dispersão (última Iteration):
        X = Individual | Y = Fitness
        Um marcador/cor por Noise (cor automática; marcador distinto)
    - Eixo Y é fixo por Model (auto com 5% de folga, ou manual via MANUAL_Y_LIMITS)
    """
    REQUIRED = ("Iteration", "Individual", "Fitness")

    def __init__(self, files: Iterable[Path], out_dir: Path):
        self.files = [Path(f) for f in files]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- utilidades de cabeçalho ----------
    @staticmethod
    def _normalize_cols(cols: Iterable[str]) -> List[str]:
        return (
            pd.Index(cols)
              .str.replace(r'^\ufeff', '', regex=True)  # remove BOM
              .str.strip()
              .tolist()
        )

    @staticmethod
    def _alias_picker(columns: List[str]) -> Dict[str, str]:
        """
        Recebe os nomes originais e retorna um mapa {alvo: nome_real}
        Aceita variações (lower-case) como 'iter', 'geracao', etc.
        """
        norm = {c.lower(): c for c in GAGroupScatter._normalize_cols(columns)}

        def pick(*aliases) -> Optional[str]:
            for a in aliases:
                if a in norm:
                    return norm[a]
            return None

        col_iteration  = pick('iteration', 'iter', 'geracao', 'geração', 'generation')
        col_individual = pick('individual', 'individuo', 'indivíduo', 'id')
        col_fitness    = pick('fitness')

        found = {}
        if col_iteration is not None:
            found['Iteration'] = col_iteration
        if col_individual is not None:
            found['Individual'] = col_individual
        if col_fitness is not None:
            found['Fitness'] = col_fitness
        return found

    def _read_last_iter(self, fmeta: FileMeta) -> Optional[pd.DataFrame]:
        """
        Lê apenas as colunas necessárias com ; e utf-8-sig, normaliza, filtra última Iteration
        e adiciona Model/Analysis/Noise/Source.
        """
        # 1) Lê só o cabeçalho para mapear aliases -> nomes reais
        try:
            header_only = pd.read_csv(
                fmeta.path, sep=';', encoding='utf-8-sig', nrows=0
            )
        except Exception as e:
            print(f"[ERRO] Falha ao ler cabeçalho {fmeta.path}: {e}")
            return None

        mapping = self._alias_picker(header_only.columns.tolist())
        missing = [t for t in self.REQUIRED if t not in mapping]
        if missing:
            print(f"[AVISO] {fmeta.path.name} ignorado: faltam colunas {set(missing)}")
            return None

        # 2) Lê apenas as colunas mapeadas, tudo como string
        usecols_real = list(mapping.values())
        try:
            df = pd.read_csv(
                fmeta.path,
                sep=';',
                encoding='utf-8-sig',
                usecols=usecols_real,
                dtype=str  # garante string → evita DtypeWarning
            )
        except Exception as e:
            print(f"[ERRO] Falha ao ler {fmeta.path} (usecols): {e}")
            return None

        # 3) Renomeia para os alvos padronizados
        rename_map = {v: k for k, v in mapping.items()}
        df.rename(columns=rename_map, inplace=True)

        # 4) Converte para numérico (valores inválidos viram NaN)
        df[['Iteration', 'Individual', 'Fitness']] = df[['Iteration', 'Individual', 'Fitness']].apply(
            pd.to_numeric, errors='coerce'
        )

        # 5) Limpa NaNs
        df = df.dropna(subset=['Iteration', 'Individual', 'Fitness'])
        if df.empty:
            print(f"[AVISO] {fmeta.path.name} vazio após limpeza.")
            return None

        # 6) Última Iteration
        last_it = int(df['Iteration'].max())
        dfl = df[df['Iteration'] == last_it].copy()
        if dfl.empty:
            print(f"[AVISO] {fmeta.path.name} sem linhas na última Iteration.")
            return None

        # 7) Metadados
        dfl["Model"] = fmeta.model
        dfl["Analysis"] = fmeta.analysis
        dfl["Noise"] = fmeta.noise
        dfl["Source"] = fmeta.path.stem
        return dfl

    def _collect_by_group(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        grouped: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

        for p in self.files:
            if not p.exists() or not p.is_file():
                print(f"[AVISO] Caminho ignorado: {p}")
                continue

            meta = NameRules.parse(p)
            if meta is None:
                print(f"[AVISO] Nome não segue padrão esperado (Model/Noise ausentes): {p.name}")
                continue

            dfl = self._read_last_iter(meta)
            if dfl is None:
                continue

            key = (meta.model, meta.analysis)
            grouped.setdefault(key, []).append(dfl)

        out: Dict[Tuple[str, str], pd.DataFrame] = {}
        for key, dfs in grouped.items():
            out[key] = pd.concat(dfs, ignore_index=True)
        return out

    @staticmethod
    def _marker_cycle(n: int) -> List[str]:
        base = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', 'H', 'd', 'p', '8']
        if n <= len(base):
            return base[:n]
        reps = (n + len(base) - 1) // len(base)
        return (base * reps)[:n]

    def _compute_y_limits_per_model(self, groups: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
        """Limites Y por Model: manual ou automáticos (5% folga) agregando todos os gráficos daquele Model."""
        ylims: Dict[str, Tuple[float, float]] = {}

        auto_min: Dict[str, float] = {}
        auto_max: Dict[str, float] = {}
        for (model, _), df in groups.items():
            if df.empty:
                continue
            cur_min = float(df["Fitness"].min())
            cur_max = float(df["Fitness"].max())
            auto_min[model] = min(cur_min, auto_min.get(model, cur_min))
            auto_max[model] = max(cur_max, auto_max.get(model, cur_max))

        for model in set(list(auto_min.keys()) + list(MANUAL_Y_LIMITS.keys())):
            if model in MANUAL_Y_LIMITS and MANUAL_Y_LIMITS[model]:
                ylims[model] = MANUAL_Y_LIMITS[model]
            else:
                if model not in auto_min:
                    continue
                lo, hi = auto_min[model], auto_max[model]
                if lo == hi:
                    pad = max(1.0, abs(lo) * 0.05)
                    ylims[model] = (lo - pad, hi + pad)
                else:
                    pad = 0.05 * (hi - lo)
                    ylims[model] = (lo - pad, hi + pad)
        return ylims

    def plot(self):
        groups = self._collect_by_group()
        if not groups:
            print("[INFO] Nenhum grupo válido encontrado.")
            return

        # Limites fixos por Model
        model_y = self._compute_y_limits_per_model(groups)

        for (model, analysis), df in groups.items():
            if df.empty:
                continue

            noises = sorted(df["Noise"].unique())
            markers = self._marker_cycle(len(noises))
            marker_map = {n: m for n, m in zip(noises, markers)}

            fig = plt.figure(figsize=(10, 6))
            ax = plt.gca()
            plt.subplots_adjust(**MARGINS)

            # Um scatter por Noise (cor automática; marcador distinto)
            for noise in noises:
                dfn = df[df["Noise"] == noise]
                label_txt = NOISE_LABELS.get(noise, f"{noise*100:.0f}%")
                ax.scatter(
                    dfn["Individual"].values,
                    dfn["Fitness"].values,
                    marker=marker_map[noise],
                    label=label_txt,
                    alpha=0.9
                )

            # Eixo Y fixo por Model
            if model in model_y:
                ax.set_ylim(model_y[model])

            ax.set_title(f"Fitness × Individual — Última Iteration\n{model} | {analysis}")
            ax.set_xlabel("Individual")
            ax.set_ylabel("Fitness")
            ax.grid(True, linestyle='--', alpha=0.4)

            # Legenda: à direita, centralizada verticalmente, fora do plot
            ax.legend(
                title="Noise",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True
            )

            # Salvar
            fname = f"{model}_{analysis}_scatter.png"
            out_path = self.out_dir / fname
            fig.savefig(out_path, dpi=200)  # margens fixas, não usar bbox_inches='tight'
            plt.close(fig)
            print(f"[OK] Figura salva: {out_path}")


if __name__ == "__main__":
    # Pasta com seus CSVs
    data_dir = Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data")
    files = sorted(data_dir.glob("*.csv"))  # ajuste o padrão se necessário
    out_dir = Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Out_Plots")

    GAGroupScatter(files=files, out_dir=out_dir).plot()
