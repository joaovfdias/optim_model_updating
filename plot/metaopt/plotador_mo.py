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
    salvar_como = f"Grafico_BO_1_Convergencia_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()


def plotar_sensibilidade_bo(df_bo, legenda=False, salvar_em=None):
    """
    Gera o Gráfico de Sensibilidade: Fitness vs Hiperparâmetro (O Efeito Funil)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    familias = ['EI', 'PI', 'LCB']

    for i, fam in enumerate(familias):
        df_fam = df_bo[df_bo['Family'] == fam].copy()

        # O eixo Y será Xi para EI/PI, e Kappa para LCB
        val_col = 'Xi' if fam in ['EI', 'PI'] else 'Kappa'

        # Plota os pontos: Eixo X = Fitness, Eixo Y = Hiperparâmetro
        # A cor (c) mostra em qual ciclo de refinamento o ponto foi testado
        scatter = axes[i].scatter(df_fam['Avg_Fit'], df_fam[val_col],
                                  c=df_fam['Iteration'], cmap='viridis',
                                  alpha=0.8, edgecolors='black', s=50)

        # Destaca o melhor ponto de todos com uma estrela vermelha
        vencedor = df_fam.loc[df_fam['Avg_Fit'].idxmin()]
        axes[i].scatter(vencedor['Avg_Fit'], vencedor[val_col],
                        color='red', marker='*', s=200, edgecolors='black', label='Melhor Global', zorder=5)

        axes[i].set_title(f"{fam}", fontweight='bold', fontsize=16)
        if i == 1: axes[i].set_xlabel("Erro Estrutural (Fitness)", fontsize=16, labelpad=10)
        axes[i].set_ylabel(f"Valor de $\\{val_col.lower()}$")

        # Inverte o eixo Y conforme sua sugestão (do maior pro menor)
        axes[i].invert_yaxis()

        # Aplica escala logarítmica apenas para o Xi (que varia em casas decimais)
        if val_col == 'Xi':
            axes[i].set_yscale('log')

        axes[i].grid(True, which="both", ls="--", alpha=0.5)
        if legenda and i == 1:
            axes[i].legend(loc='upper right')

    # # Adiciona a barra de cores geral da figura
    # cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), pad=0.02)
    # cbar.set_label('Ciclo de Refinamento (Iteração)')

    fig.suptitle("Efeito do Hiperparâmetro na Convergência", fontsize=18, fontweight='bold',
                 y=0.99)

    # Ajusta o layout para não encavalar
    plt.tight_layout()
    salvar_como = f"Grafico_BO_2_Sensibilidade_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()



def plotar_aprendizado_pop(df_ga, algo="GA", xticks:list=None, salvar_em=None):
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
            color='purple', linewidth=2.5, label='Melhor cumulativo')

    ax.scatter(df_ga['Iteracao'], df_ga['Score'],
               color='gray', alpha=0.6, label='Amostragem')

    ax.set_title(f"Evolução da Meta-Otimização do {algo}", fontweight='bold')
    ax.set_xlabel("Avaliações (chamadas do BO)")
    ax.set_ylabel("Pontuação $J$")

    # Ajusta o eixo X para mostrar números inteiros
    if xticks:
        ax.set_xticks(range(xticks[0], len(df_ga) + 1, xticks[1]))
    else:
        ax.set_xticks(range(1, len(df_ga) + 1, max(1, len(df_ga) // 10)))

    ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    salvar_como = f"Grafico_{algo}_1_Aprendizado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()


def plotar_sensibilidade_pop(df_pop, algo="GA", salvar_em=None):
    pass


    salvar_como = f"Grafico_{algo}_2_Sensibilidade_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()


def plotar_mapa_calor_ga(df_ga, salvar_em=None):
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
    cbar.set_label('Pontuação $J$ (Menor é Melhor)')

    # Destaca o vencedor global com uma estrela vermelha
    vencedor = df_ga.loc[df_ga['Score'].idxmin()]
    ax.scatter(vencedor['crossover_rate'], vencedor['mutation_strength'],
               color='red', marker='*', s=300, label='Configuração ótima', edgecolors='black')

    ax.set_title("Espaço de Hiperparâmetros do GA", fontweight='bold')
    ax.set_xlabel("Taxa de Crossover")
    ax.set_ylabel("Força de Mutação")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc='lower left')

    salvar_como = f"Grafico_GA_3_MapaCalor.png_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
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