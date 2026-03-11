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
        if i == 1: axes[i].set_xlabel("Fitness", fontsize=16, labelpad=10)
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


def plotar_sensibilidade_bo_unificado(df_bo, salvar_em=None):
    """
    Gera o Gráfico de Sensibilidade Unificado para todas as funções de aquisição.
    Eixo Y Esq: Kappa (Linear) | Eixo Y Dir: Xi (Logarítmico)
    """
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Eixo Y secundário para o Xi (que divide o mesmo Eixo X)
    ax2 = ax1.twinx()

    # Separa os DataFrames
    df_ei = df_bo[df_bo['Family'] == 'EI'].copy()
    df_pi = df_bo[df_bo['Family'] == 'PI'].copy()
    df_lcb = df_bo[df_bo['Family'] == 'LCB'].copy()

    # 1. Plota LCB no eixo esquerdo (ax1) - Kappa (Vermelho)
    ax1.scatter(df_lcb['Avg_Fit'], df_lcb['Kappa'], color='red', alpha=0.6, edgecolors='black', s=60,
                label='LCB ($\kappa$)')
    best_lcb = df_lcb.loc[df_lcb['Avg_Fit'].idxmin()]
    ax1.scatter(best_lcb['Avg_Fit'], best_lcb['Kappa'], color='darkred', marker='*', s=400, edgecolors='black',
                zorder=5)

    # 2. Plota EI e PI no eixo direito (ax2) - Xi (Azul e Verde)
    ax2.scatter(df_ei['Avg_Fit'], df_ei['Xi'], color='royalblue', alpha=0.6, edgecolors='black', s=60,
                label='EI ($\\xi$)')
    best_ei = df_ei.loc[df_ei['Avg_Fit'].idxmin()]
    ax2.scatter(best_ei['Avg_Fit'], best_ei['Xi'], color='blue', marker='*', s=400, edgecolors='black', zorder=5)

    ax2.scatter(df_pi['Avg_Fit'], df_pi['Xi'], color='limegreen', alpha=0.6, edgecolors='black', s=60,
                label='PI ($\\xi$)')
    best_pi = df_pi.loc[df_pi['Avg_Fit'].idxmin()]
    ax2.scatter(best_pi['Avg_Fit'], best_pi['Xi'], color='green', marker='*', s=400, edgecolors='black', zorder=5)

    # --- FORMATAÇÃO DOS EIXOS ---

    # Eixo X (Fitness)
    ax1.set_xlabel("Fitness", fontweight='bold', fontsize=14, labelpad=15)
    ax1.invert_xaxis()  # DECRESCENTE: Ponto cego na esquerda, precisão na direita

    # Eixo Y Esquerdo (Kappa)
    ax1.set_ylabel("Valor de $\kappa$ (LCB)", fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')

    # Eixo Y Direito (Xi)
    ax2.set_ylabel("Valor de $\\xi$ (EI e PI)", fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_yscale('log')  # Escala Logarítmica para o Xi

    # Opcional: Se quiser inverter também os eixos Y (deixar os maiores valores para baixo)
    # basta descomentar as duas linhas abaixo:
    # ax1.invert_yaxis()
    # ax2.invert_yaxis()

    # Adiciona o grid tracejado no eixo principal
    ax1.grid(True, which='both', ls='--', alpha=0.5)

    # Junta as legendas dos dois eixos para ficar num quadro só
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # A legenda fica na esquerda (onde o erro é alto) para não cobrir as estrelas (onde o erro é baixo)
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)

    plt.title("Desempenho das Funções de Aquisição", fontweight='bold', fontsize=16, pad=10)

    # Ajusta as margens para que os dois eixos Y apareçam perfeitamente
    plt.tight_layout()
    salvar_como = f"Grafico_BO_3_Sensibilidade_Unificado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {salvar_em}")
    plt.show()
    plt.close()


def plotar_aprendizado_pop(df_pop, algo="GA", xticks:list=None, salvar_em=None):
    """
    Gera o Gráfico 2: Curva de aprendizado do BO otimizando o GA
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Cria a coluna de Iteração (sequencial, assumindo que o log está em ordem)
    df_ga = df_pop.sort_values('Timestamp').reset_index(drop=True)
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


def plotar_correlacao_pop(df_pop, algo="GA", salvar_em=None):

    df = df_pop.copy()

    # Converter fitness real (opcional)
    df["Avg_Fit"] = np.exp(df["Avg_LogFit"])

    # Seleção automática dos hiperparâmetros
    if algo.upper() == "GA":
        rows = ["elitism_rate", "crossover_rate", "mutation_strength"]
    elif algo.upper() == "PSO":
        rows = ["w", "w_rate", "c1", "c2", "init_vel_ratio"]
    else:
        raise ValueError("Algoritmo deve ser 'GA' ou 'PSO'")

    cols = ["Avg_Fit"]

    corr = df[rows + cols].corr().loc[rows, cols]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    # valores numéricos
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Correlation")

    plt.title(f"Correlation of {algo} Hyperparameters with Fitness")

    plt.tight_layout()

    salvar_como = f"Grafico_{algo}_2_Correlacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    if salvar_em:
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {os.path.join(salvar_em, salvar_como)}")

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


def plotar_mapa_calor_pso_cognitivo_social(df_pso, salvar_em=None):
    """
    Gera o Mapa de Dispersão Cognitivo (c1) vs Social (c2).
    Mostra se o enxame foi mais explorador (c2 > c1) ou intensificador (c1 > c2).
    """
    # Filtra as falhas numéricas (onde o Score foi 1.0) para não estragar a escala de cores
    df_valido = df_pso[df_pso['Score'] < 1.0].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Vamos inverter o Score para que "Maior" bolinha = Melhor (Menor J)
    tamanho = (df_valido['Score'].max() - df_valido['Score'] + 0.01) * 1000

    # Plota o Scatter: Eixo X = c1, Eixo Y = c2
    scatter = ax.scatter(df_valido['c1'], df_valido['c2'],
                         c=df_valido['Score'], cmap='viridis_r', s=tamanho, alpha=0.8, edgecolors='black')

    # Adiciona a barra de cores
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Pontuação $J$ (Menor é Melhor)')

    # Destaca o vencedor global com uma estrela vermelha
    vencedor = df_valido.loc[df_valido['Score'].idxmin()]
    ax.scatter(vencedor['c1'], vencedor['c2'],
               color='red', marker='*', s=300, label='Configuração ótima', edgecolors='black', zorder=5)

    ax.set_title("Espaço de Hiperparâmetros do PSO: Cognitivo x Social", fontweight='bold')
    ax.set_xlabel("Coeficiente Cognitivo ($c_1$)")
    ax.set_ylabel("Coeficiente Social ($c_2$)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc='lower left')

    salvar_como = f"Grafico_PSO_2_CognitivoSocial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if salvar_em:
        # Garante que a pasta existe antes de salvar
        os.makedirs(salvar_em, exist_ok=True)
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {os.path.join(salvar_em, salvar_como)}")

    plt.tight_layout()
    plt.show()
    plt.close()


def plotar_mapa_calor_pso_inercia_atracao(df_pso, salvar_em=None):
    """
    Gera o Mapa de Estabilidade Inércia (w) vs Força de Atração (c1 + c2).
    """
    # Filtra as falhas
    df_valido = df_pso[df_pso['Score'] < 1.0].copy()

    # Cria a coluna da Força de Atração (Soma de c1 e c2)
    df_valido['forca_atracao'] = df_valido['c1'] + df_valido['c2']

    fig, ax = plt.subplots(figsize=(8, 6))

    tamanho = (df_valido['Score'].max() - df_valido['Score'] + 0.01) * 1000

    # Plota o Scatter: Eixo X = w, Eixo Y = c1 + c2
    scatter = ax.scatter(df_valido['w'], df_valido['forca_atracao'],
                         c=df_valido['Score'], cmap='viridis_r', s=tamanho, alpha=0.8, edgecolors='black')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Pontuação $J$ (Menor é Melhor)')

    vencedor = df_valido.loc[df_valido['Score'].idxmin()]
    ax.scatter(vencedor['w'], vencedor['forca_atracao'],
               color='red', marker='*', s=300, label='Configuração ótima', edgecolors='black', zorder=5)

    ax.set_title("Estabilidade do PSO: Inércia x Força de Atração", fontweight='bold')
    ax.set_xlabel("Inércia ($w$)")
    ax.set_ylabel("Força de Atração Total ($c_1 + c_2$)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc='lower left')

    salvar_como = f"Grafico_PSO_3_InerciaAtracao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)
        fig.savefig(os.path.join(salvar_em, salvar_como), dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: {os.path.join(salvar_em, salvar_como)}")

    plt.tight_layout()
    plt.show()
    plt.close()
