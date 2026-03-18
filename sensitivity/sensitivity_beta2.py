from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Heatmap (pip install seaborn)
from scipy.stats import spearmanr

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


# Estruturas de Dados

@dataclass
class ParamSpec:
    name: str
    lower: float
    upper: float
    kind: Literal["continuous", "integer"] = "continuous"


# Amostragem

class Sampler:

    @staticmethod
    def random(n: int, specs: Sequence[ParamSpec], rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            row = {}
            for p in specs:
                v = rng.uniform(p.lower, p.upper)
                row[p.name] = int(round(v)) if p.kind == "integer" else float(v)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def lhs(n: int, specs: Sequence[ParamSpec], rng: np.random.Generator) -> pd.DataFrame:
        """Latin Hypercube Sampling"""
        if n <= 0: raise ValueError("n deve ser positivo")
        cols = {}
        for p in specs:
            cut = np.linspace(0.0, 1.0, n + 1)
            u = rng.random(n)
            pts = cut[:-1] + u * (1.0 / n)
            rng.shuffle(pts)
            vals = p.lower + pts * (p.upper - p.lower)
            vals = np.rint(vals).astype(int) if p.kind == "integer" else vals.astype(float)
            cols[p.name] = vals
        return pd.DataFrame(cols)

    @staticmethod
    def generate_morris_trajectories(n_trajectories: int, specs: Sequence[ParamSpec],
                                     num_levels: int = 4) -> pd.DataFrame:
        """Gera trajetórias para o Metodo de Morris (Screening)."""

        problem = {
            'num_vars': len(specs),
            'names': [p.name for p in specs],
            'bounds': [[p.lower, p.upper] for p in specs]
        }
        X = morris_sample.sample(problem, N=n_trajectories, num_levels=num_levels)
        df = pd.DataFrame(X, columns=[p.name for p in specs])
        for p in specs:
            if p.kind == "integer":
                df[p.name] = df[p.name].round().astype(int)
        return df


# Módulo de Análise

class SensitivityAnalyzer:
    """Realiza a análise estatística."""

    @staticmethod
    def spearman_simple(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Cálculo simples de Spearman (Matriz de Correlação)."""
        # Junta X e y para calcular matriz completa
        data = X.copy()
        data['Target'] = y

        # Calcula matriz de correlação
        corr_matrix = data.corr(method='spearman')

        # Extrai apenas a correlação com o Target
        target_corr = corr_matrix['Target'].drop('Target')

        df_res = pd.DataFrame({
            'Parameter': target_corr.index,
            'Spearman_Rho': target_corr.values,
            'Abs_Rho': np.abs(target_corr.values)
        }).sort_values('Abs_Rho', ascending=False)

        return df_res, corr_matrix

    @staticmethod
    def spearman_bootstrap(X: pd.DataFrame, y: pd.Series, n_boot: int = 1000,
                           confidence: float = 0.95, rng_seed: int = 42) -> pd.DataFrame:
        """Spearman com Intervalo de Confiança (Robusto)."""
        rng = np.random.default_rng(rng_seed)
        n_samples = len(y)
        results = []
        alpha = (1.0 - confidence) / 2.0
        q_lower, q_upper = alpha * 100, (1.0 - alpha) * 100

        print(f"\n[Bootstrap] Executando {n_boot} reamostragens para {len(X.columns)} parâmetros...")

        for param in X.columns:
            rhos = []
            x_data, y_data = X[param].values, y.values
            for _ in range(n_boot):
                idx = rng.choice(n_samples, n_samples, replace=True)
                # Proteção contra desvio padrão zero
                if np.std(x_data[idx]) == 0 or np.std(y_data[idx]) == 0:
                    rhos.append(0.0)
                else:
                    rho, _ = spearmanr(x_data[idx], y_data[idx])
                    rhos.append(0.0 if np.isnan(rho) else rho)

            mean_rho = np.mean(rhos)
            ci_lower, ci_upper = np.percentile(rhos, q_lower), np.percentile(rhos, q_upper)

            # Decisão: Se cruzar zero, é descartável
            zero_crossed = (ci_lower < 0 < ci_upper)
            status = "DESCARTAR" if zero_crossed else "MANTER"

            results.append({
                'Parameter': param,
                'Mean_Rho': mean_rho,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Status': status
            })

        return pd.DataFrame(results).sort_values('Mean_Rho', key=abs, ascending=False)

    @staticmethod
    def morris_screening(X: pd.DataFrame, y: pd.Series, specs: Sequence[ParamSpec]) -> pd.DataFrame:
        """Calcula índices de Morris."""
        problem = {'num_vars': len(specs), 'names': [p.name for p in specs],
                   'bounds': [[p.lower, p.upper] for p in specs]}
        Si = morris_analyze.analyze(problem, X.to_numpy(), y.to_numpy(), conf_level=0.95)
        return pd.DataFrame({
            'Parameter': problem['names'],
            'Mu_Star': Si['mu_star'],
            'Sigma': Si['sigma']
        }).sort_values('Mu_Star', ascending=False)


# Visualização e Relatório

class SensitivityReporter:

    @staticmethod
    def plot_heatmap(corr_matrix: pd.DataFrame, title: str = "Matriz de Correlação", save_path: str = None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)  # Salva se um caminho for passado
            print(f"[Plot] Heatmap salvo em: {save_path}")

        plt.show()

    @staticmethod
    def plot_bootstrap_intervals(df_boot: pd.DataFrame):
        """Plota barras de erro do Bootstrap."""
        df_plot = df_boot.sort_values('Mean_Rho', ascending=True)
        colors = ['red' if s == 'DESCARTAR' else 'green' for s in df_plot['Status']]
        xerr = [df_plot['Mean_Rho'] - df_plot['CI_Lower'], df_plot['CI_Upper'] - df_plot['Mean_Rho']]

        plt.figure(figsize=(10, 6))
        plt.errorbar(df_plot['Mean_Rho'], range(len(df_plot)), xerr=xerr, fmt='o', color='black', ecolor=colors,
                     capsize=5)
        plt.axvline(0, color='gray', linestyle='--')
        plt.yticks(range(len(df_plot)), df_plot['Parameter'])
        plt.title("Sensibilidade Robusta (95% IC)")
        plt.xlabel("Coeficiente de Spearman")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_morris(df_morris: pd.DataFrame):
        """Plota gráfico de Morris (Mu* vs Sigma)."""
        plt.figure(figsize=(10, 6))
        plt.scatter(df_morris['Mu_Star'], df_morris['Sigma'])
        for _, row in df_morris.iterrows():
            plt.text(row['Mu_Star'], row['Sigma'], row['Parameter'])
        plt.xlabel("μ* (Influência Total)")
        plt.ylabel("σ (Interação)")
        plt.title("Método de Morris")
        plt.grid(True)
        plt.show()

    @staticmethod
    def save_results(df: pd.DataFrame, filename_base: str, output_dir: str = "."):
        """Salva resultados em CSV e TXT."""
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # 1. Salva CSV (Dados brutos)
        csv_path = os.path.join(output_dir, f"{filename_base}.csv")
        df.to_csv(csv_path, index=False, sep=';', decimal=',')

        # 2. Salva TXT (Relatório legível)
        txt_path = os.path.join(output_dir, f"{filename_base}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== RELATÓRIO DE SENSIBILIDADE: {filename_base} ===\n")
            f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(df.to_string())

        print(f"[Salvo] Resultados exportados para:\n - {csv_path}\n - {txt_path}")

    @staticmethod
    def print_ranking_terminal(df: pd.DataFrame, metric_col: str):
        """Imprime o ranking formatado no terminal."""
        print(f"\n{'=' * 40}")
        print(f"RANKING DE SENSIBILIDADE ({metric_col})")
        print(f"{'=' * 40}")
        print(df.to_string(index=False))
        print(f"{'=' * 40}\n")


# Exemplo de Fluxo

if __name__ == "__main__":
    # 1. Definição do Problema
    specs = [
        ParamSpec("E_Viga", 20e9, 35e9),
        ParamSpec("K_Mola", 1e6, 1e8),
        ParamSpec("Densidade", 2000, 2500),
        ParamSpec("Dummy_Noise", 0, 1)  # Variável inútil
    ]

    # 2. Amostragem (LHS para Spearman)
    print(">>> Gerando Amostras...")
    rng = np.random.default_rng()
    X = Sampler.lhs(n=50, specs=specs, rng=rng)

    # Simulação do Modelo (Substitua pelo seu loop do ANSYS)
    # y = f(X) -> Aqui simulado: E_Viga é forte, K_Mola é médio, Dummy é nulo
    y = (X["E_Viga"] * 0.8) + (X["K_Mola"] * 0.3) + np.random.normal(0, 1e9, 50)

    # 3. Análise BOOTSTRAP (A mais importante)
    df_boot = SensitivityAnalyzer.spearman_bootstrap(X, y)

    # 4. Resultados: Terminal, Gráficos e Arquivos
    SensitivityReporter.print_ranking_terminal(df_boot, "Mean_Rho")
    SensitivityReporter.save_results(df_boot, "resultado_sensibilidade_bootstrap")

    # Gráficos
    SensitivityReporter.plot_bootstrap_intervals(df_boot)

    # Opcional: Heatmap da matriz completa
    _, corr_matrix = SensitivityAnalyzer.spearman_simple(X, y)
    SensitivityReporter.plot_heatmap(corr_matrix)