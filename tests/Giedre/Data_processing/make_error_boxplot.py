# -*- coding: utf-8 -*-
"""
Gera boxplots por Model × coluna de erro a partir de Final_Data.csv
- CSV com separador ';' e vírgula como decimal
- Cada box = valores (trials) de um grupo (Model, Analysis, Noise)
- X = Noise | Y = Erro (%) | cores = Analysis
- Saída: plots_percent/<Model>/<Model>__<Erro>.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================

CSV_PATH = Path(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Final_data\Final_Data.csv')          # ajuste se necessário
OUT_DIR  = Path(r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Out_Plots")          # pasta-base de saída
SHOW     = True                          # True para abrir as figuras na tela

# Se quiser limitar quais colunas de erro plotar, liste aqui.
# Caso deixe como None, o script detecta automaticamente todas que contenham "Error".
ERROR_COLS_MANUAL = None
# ERROR_COLS_MANUAL = ["E Error", "ν Error", "r Error", "k1 Error", "k2 Error", "k3 Error", "k4 Error"]

# =========================
# FUNÇÕES AUXILIARES
# =========================
def read_csv_semicolon(path: Path) -> pd.DataFrame:
    """Lê o CSV como texto (dtype=str) para fazer limpeza robusta depois."""
    df = None
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(path, sep=";", encoding=enc, engine="python", dtype=str)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError("Falha ao ler o CSV com sep=';'. Verifique caminho/arquivo.")
    # normalizar cabeçalhos e strings-chave
    df.columns = [c.strip() for c in df.columns]
    for key in ("Model", "Analysis", "Noise"):
        if key in df.columns:
            df[key] = df[key].astype(str).str.strip()
    return df

def clean_number_series(s: pd.Series) -> pd.Series:
    """
    Limpa séries numéricas em formato 'pt-BR':
    - remove espaços e %,
    - troca vírgula por ponto,
    - remove quaisquer caracteres não numéricos (exceto sinal, ponto e expoente),
    - converte para float (coerce -> NaN quando não possível).
    """
    s = s.astype(str)
    s = s.str.replace("\xa0", " ", regex=False).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.apply(lambda x: re.sub(r"[^0-9eE\.\+\-]", "", x))
    return pd.to_numeric(s, errors="coerce")

def detect_error_cols(df: pd.DataFrame) -> list[str]:
    if ERROR_COLS_MANUAL:
        cols = [c for c in ERROR_COLS_MANUAL if c in df.columns]
    else:
        cols = [c for c in df.columns if "Error" in str(c)]
    if not cols:
        raise ValueError("Nenhuma coluna de erro encontrada (nome contendo 'Error').")
    return cols

def prepare_levels(df: pd.DataFrame):
    """
    Define níveis de Noise (numérico se possível) e de Analysis,
    e retorna (df_mod, use_noise_num, noise_levels, xticklabels, analyses)
    """
    noise_num = clean_number_series(df["Noise"])
    use_noise_num = noise_num.notna().sum() >= df["Noise"].notna().sum() * 0.5
    if use_noise_num:
        df = df.copy()
        df["Noise_num"] = noise_num
        noise_levels = sorted(df["Noise_num"].dropna().unique())
        xticklabels = [str(n) for n in noise_levels]
    else:
        noise_levels = sorted(df["Noise"].dropna().astype(str).unique())
        xticklabels = [str(n) for n in noise_levels]
    analyses = sorted(df["Analysis"].dropna().astype(str).unique())
    return df, use_noise_num, noise_levels, xticklabels, analyses

def plot_model_error(df_model: pd.DataFrame, err_col: str, out_path: Path, show: bool=False) -> bool:
    """
    Gera o boxplot de um 'err_col' para um 'Model' (df_model já filtrado).
    Retorna True se salvou (tinha dados), False se não havia dados válidos.
    """
    # Configuração de fonte e margens fixas
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.subplot.left'] = 0.1
    plt.rcParams['figure.subplot.right'] = 0.8
    plt.rcParams['figure.subplot.top'] = 0.9
    plt.rcParams['figure.subplot.bottom'] = 0.1

    df_m, use_noise_num, noise_levels, xticklabels, analyses = prepare_levels(df_model)

    base_positions = np.arange(1, len(noise_levels) + 1)
    n_a = max(1, len(analyses))
    group_w = 0.7
    box_w = group_w / n_a
    offsets = (np.linspace(-(group_w/2)+box_w/2, (group_w/2)-box_w/2, n_a)
               if n_a > 1 else np.array([0.0]))

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [None]*n_a
    handles, labels = [], []

    any_data = False
    for i, a in enumerate(analyses):
        data = []
        for nl in noise_levels:
            if use_noise_num:
                mask = (df_m["Analysis"] == a) & (df_m["Noise_num"] == nl)
            else:
                mask = (df_m["Analysis"] == a) & (df_m["Noise"] == nl)
            vals = pd.to_numeric(df_m.loc[mask, err_col], errors="coerce").dropna().values
            data.append(vals)
            if len(vals) > 0:
                any_data = True

        positions = base_positions + offsets[i]
        bp = ax.boxplot(data, positions=positions, widths=box_w*0.9,
                        patch_artist=True, manage_ticks=False)

        this_color = color_cycle[i % len(color_cycle)]
        for patch in bp["boxes"]:
            if this_color is not None:
                patch.set_facecolor(this_color)
        for k in ["whiskers", "caps", "medians", "fliers"]:
            for item in bp[k]:
                if this_color is not None:
                    try:
                        item.set_color(this_color)
                    except Exception:
                        pass

        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(facecolor=this_color if this_color else "white",
                                      edgecolor="black"))
        labels.append(str(a))

    # Título e eixos
    title_model = str(df_model["Model"].iloc[0]) if "Model" in df_model.columns and not df_model.empty else ""
    title = f"{title_model} — {err_col}"
    ax.set_title(title)
    ax.set_xlabel("Noise")
    ax.set_ylabel(f"{err_col} = value / reference value")
    ax.set_xticks(base_positions)
    ax.set_xticklabels(xticklabels)
    # Fixando os ranges
    plt.ylim(0, 3.5)  # Limites no eixo Y
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Legenda no centro à direita, fora do gráfico
    if handles:
        ax.legend(handles, labels, title="Analysis", frameon=True,
                  loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

    if any_data:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        if show:
            plt.show()
    plt.close(fig)
    return any_data

# =========================
# PIPELINE PRINCIPAL
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = read_csv_semicolon(CSV_PATH)

    # Detectar colunas de erro e converter todas para numérico (em % já "limpo")
    error_cols = detect_error_cols(df_raw)
    df = df_raw.copy()
    for c in error_cols:
        df[c] = clean_number_series(df[c])

    # Iterar por modelo e por coluna de erro
    models = sorted(df["Model"].dropna().unique(), key=str) if "Model" in df.columns else []
    total_imgs = 0

    for model in models:
        df_m = df[df["Model"] == model].copy()
        # Pula se faltar base mínima
        if df_m.empty or df_m["Analysis"].isna().all() or df_m["Noise"].isna().all():
            continue

        model_dir = OUT_DIR / str(model)
        for err_col in error_cols:
            out_file = model_dir / f"{str(model)}__{err_col.replace(' ', '_')}.png"
            ok = plot_model_error(df_m, err_col, out_file, show=SHOW)
            if ok:
                total_imgs += 1
            else:
                print(f"[AVISO] Sem dados válidos para: Model={model}, Error={err_col}")

    print(f"Concluído. Imagens geradas: {total_imgs}. Pasta: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
