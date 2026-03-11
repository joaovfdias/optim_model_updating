import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURAÇÃO GLOBAL ACADÊMICA ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def plotar_convergencia_bo(df_bo, ylim:dict={}, log:bool=False, salvar_em=None):
    """
    Gera o Gráfico 1: O 'Zoom' Logarítmico do BO (Refinamento de Xi e Kappa)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    familias = ['EI', 'PI', 'LCB']

    for i, fam in enumerate(familias):
        df_fam = df_bo[df_bo['Family'] == fam]

        # Pega a melhor configuração (menor Score) de cada iteração para traçar a linha do ótimo
        # O Score no CSV do BO é local (por iteração), então o min() é o centro do próximo grid
        idx_melhores = df_fam.groupby('Iteration')['Score'].idxmin()
        best_path = df_fam.loc[idx_melhores].sort_values('Iteration')

        val_col = 'Xi' if fam in ['EI', 'PI'] else 'Kappa'
        cor = 'blue' if fam == 'EI' else ('green' if fam == 'PI' else 'red')

        # Plota a trajetória do Hiperparâmetro Ótimo
        axes[i].plot(best_path['Iteration'], best_path[val_col], marker='o',
                     linestyle='-', linewidth=2, color=cor, label=f'Trajetória Ótima de $\\{val_col.lower()}$')

        # Plota todos os pontos testados no fundo para mostrar o "Grid Search" encolhendo
        axes[i].scatter(df_fam['Iteration'], df_fam[val_col], color='gray', alpha=0.3, s=20, label='Pontos Avaliados')

        axes[i].set_title(f"Refinamento Logarítmico - Função {fam}", fontweight='bold')
        axes[i].set_ylabel(f"Valor de $\\{val_col.lower()}$")
        if log: axes[i].set_yscale('log')
        if fam in list(ylim.keys()):
            axes[i].set_ylim(None, ylim[fam])

        axes[i].grid(True, which="both", ls="--", alpha=0.5)
        axes[i].legend(loc='best', fontsize=10)

    axes[-1].set_xlabel("Ciclo de Refinamento (Iteração)")

    fig.tight_layout()
    salvar_como = f"Grafico1_BO_Convergencia_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em: fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()


def plotar_aprendizado_ga(df_ga, salvar_em=None):
    """
    Gera o Gráfico 2: Curva de aprendizado do BO otimizando o GA
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Cria a coluna de Iteração (sequencial, assumindo que o log está em ordem)
    df_ga = df_ga.sort_values('Timestamp').reset_index(drop=True)
    df_ga['Iteracao'] = df_ga.index + 1

    # Calcula a "Melhor Pontuação Encontrada Até o Momento" (CumMin)
    df_ga['Melhor_Score_Acumulado'] = df_ga['Score'].cummin()

    ax.plot(df_ga['Iteracao'], df_ga['Melhor_Score_Acumulado'],
            color='purple', linewidth=2.5, label='Melhor Pontuação (Acumulada)')

    ax.scatter(df_ga['Iteracao'], df_ga['Score'],
               color='gray', alpha=0.6, label='Tentativas do Meta-BO')

    ax.set_title("Evolução da Meta-Otimização do Algoritmo Genético", fontweight='bold')
    ax.set_xlabel("Avaliações (Chamadas do BO)")
    ax.set_ylabel("Pontuação Multicritério ($J$)")

    # Ajusta o eixo X para mostrar números inteiros
    ax.set_xticks(range(1, len(df_ga) + 1, max(1, len(df_ga) // 10)))

    ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    # fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
    # print(f"[OK] Salvo: {salvar_como}")
    # plt.close()
    plt.show()

def plotar_mapa_calor_ga(df_ga, salvar_como="Grafico3_GA_MapaCalor.png"):
    """
    Gera o Gráfico 3: Mapa de Dispersão Crossover vs Mutação (Onde o GA é melhor?)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Vamos inverter o Score para que "Maior" bolinha = Melhor (Menor J)
    # Apenas para fins de visualização do tamanho
    tamanho = (df_ga['Score'].max() - df_ga['Score'] + 0.01) * 1000

    # Plota o Scatter
    scatter = ax.scatter(df_ga['crossover_rate'], df_ga['mutation_strength'],
                         c=df_ga['Score'], cmap='viridis_r', s=tamanho, alpha=0.8, edgecolors='black')

    # Adiciona a barra de cores
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Pontuação Final $J$ (Menor é Melhor)')

    # Destaca o vencedor global com uma estrela vermelha
    vencedor = df_ga.loc[df_ga['Score'].idxmin()]
    ax.scatter(vencedor['crossover_rate'], vencedor['mutation_strength'],
               color='red', marker='*', s=300, label='Configuração Ótima', edgecolors='black')

    ax.set_title("Espaço de Hiperparâmetros Ótimos do GA", fontweight='bold')
    ax.set_xlabel("Taxa de Crossover")
    ax.set_ylabel("Força de Mutação")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar_como}")
    plt.close()


if __name__ == '__main__':

    BO_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 2\META-OPT\6 PARAM\BO"

    GA_dir = r""
    PSO_dir = r""

    save_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 2\Plotagens"

    # Carrega os Logs
    df_bo = pd.read_csv(os.path.join(BO_dir, "meta_opt_BO.csv"))
    # df_ga = pd.read_csv("meta_opt_GA.csv")

    print("Gerando os gráficos para a Dissertação...")
    plotar_convergencia_bo(df_bo, save_dir)
    # plotar_aprendizado_ga(df_ga)
    # plotar_mapa_calor_ga(df_ga)
    print("Concluído!")