import os
import time
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Queue

# --- Importações do seu pacote ---
from optimization.parameter import Continuous
from external.parser import Ansys
from external.special_functions import SpecialFun
from optimization.bo_optimizer.bayesian_from_skopt import BO


# 1. CONFIGURAÇÕES GLOBAIS


# Caminhos
ANSYS_EXE_PATH = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
# Pasta FIXA onde o ModBase.db já deve estar (sem subpastas)
BASE_DIR = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"

ANSYS_WORKING_DIR = os.path.join(BASE_DIR, '../ANSYS')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# Configuração da Otimização
N_REPETICOES = 5  # Rodadas por configuração para média estatística
N_INITIAL_POINTS = 45  # Pontos Iniciais
N_EVALUATIONS = 150  # Total (45 iniciais + 105 iterações do BO)

# Critérios de Parada do Refinamento
MIN_PARAM_DELTA = 1e-3  # Se a mudança no Xi/Kappa for menor que isso, para.
MIN_SCORE_IMPROVEMENT = 1e-4  # Se o score não melhorar isso, para.

# Pesos para o Ranking (Score)
W_FITNESS = 0.6
W_TIME = 0.2
W_STD = 0.2



# 2. WORKER (Execução Isolada)

def worker_optimization(config, run_id, result_queue):
    """
    Roda uma otimização completa em um processo isolado, usando sempre
    a mesma pasta de trabalho (ANSYS_WORKING_DIR).
    """
    try:
        # A. Definição dos Parâmetros (Laje)
        parameters = [
            Continuous(20e9, 35e9, 'modulo_viga_1'),
            Continuous(20e9, 35e9, 'modulo_viga_2'),
            Continuous(20e9, 35e9, 'modulo_centro'),
            Continuous(20e9, 35e9, 'modulo_borda_1'),
            Continuous(20e9, 35e9, 'modulo_borda_2'),
            Continuous(50e6, 50e8, 'rigidez1'),
            Continuous(50e6, 50e8, 'rigidez2'),
            Continuous(50e6, 50e8, 'rigidez3'),
            Continuous(50e6, 50e8, 'rigidez4')
        ]
        keys = [p.key for p in parameters]

        # B. Setup do ANSYS (Pasta Fixa)

        ansys = Ansys(
            ANSYS_EXE_PATH,
            ANSYS_WORKING_DIR,
            INPUT_DIR,
            "script problema 2.mac",
            "target_freq.txt",
            "target_modes.txt",
            OUTPUT_DIR
        )
        ansys.set_output_filenames("out_freq.txt", "out_modes.txt")
        ansys.max_attempts = 6

        # C. Função Fitness
        def fitness_function(param):
            input_file = ansys.create_input_file(param, keys)
            ansys.run_ansys(input_file, True, True)

            try:
                comp_freq = ansys.read_frequencies()
                comp_modes = ansys.read_modes()

                paired_freq, paired_modes, mac_err = SpecialFun.pair_modes_mac(
                    comp_freq, comp_modes, ansys.base_modes
                )
                freq_err = SpecialFun.norm_freq_errors(ansys.base_freq, paired_freq)

                fitness = (1.0 * freq_err) + (1.0 * mac_err)
                return fitness, {"freq_err": freq_err, "mac_err": mac_err, "Freq.": paired_freq, "Mode": paired_modes}
            except Exception as e:
                return 1e6, {}  # Penalidade por falha

        # D. Otimizador
        log_dir = os.path.join(os.getcwd(), '../auto_meta_logs')
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        optimizer = BO(fitness_function, parameters, N_INITIAL_POINTS)
        optimizer.set_sampling_method('lhs')
        # Sincronia de tempo não é crítica aqui, mas mantemos

        # Nome único para o log deste processo
        log_name = f"{config['tag']}_R{run_id}_{datetime.now().strftime('%H%M%S')}"
        optimizer.set_log(log_name, log_dir)

        start_t = time.time()

        # E. Execução
        result = optimizer.run(
            evaluations=N_EVALUATIONS,
            acq_func=config['acq_func'],
            xi=config['xi'],
            kappa=config['kappa'],
            status=False,
            log=True
        )

        elapsed = time.time() - start_t

        # F. Retorno
        result_queue.put({'fitness': result.fun, 'time': elapsed, 'success': True})

        # Limpeza Final (Force Kill ANSYS para liberar .lock da pasta)
        try:
            ansys.mapdl.exit()
        except:
            pass

    except Exception as e:
        print(f"[ERRO WORKER] {e}")
        result_queue.put({'fitness': 1e6, 'time': 0, 'success': False})



# 3. LÓGICA DE REFINAMENTO E SCORE


def calculate_scores(df):
    """Calcula score normalizado para comparar configs."""

    # Normalização segura (evita div por 0)
    def norm(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else pd.Series([0] * len(s))

    df['CV_Fit'] = df['Std_Fit'] / (df['Avg_Fit'].abs() + 1e-12)

    df['Norm_Fit'] = norm(df['Avg_Fit'])
    df['Norm_Time'] = norm(df['Avg_Time'])
    df['Norm_CV'] = norm(df['CV_Fit'])  # Normaliza o CV, não o Std absoluto

    # Score: Menor é melhor
    df['Score'] = (W_FITNESS * df['Norm_Fit'] +
                   W_TIME * df['Norm_Time'] +
                   W_STD * df['Norm_CV'])  # Usa o CV normalizado

    return df.sort_values('Score')


def generate_refined_grid(best_config, delta_log=0.5):  # delta_log=0.3 ~= dobra/divide por 2
    """
    Gera novos pontos usando escala LOGARÍTMICA.
    Candidatos: [10^(log-delta), Valor_Atual, 10^(log+delta)]
    """
    acq = best_config['Acq']
    current_val = best_config['Xi'] if acq in ['EI', 'PI'] else best_config['Kappa']

    # --- ALTERAÇÃO 2: Refinamento Logarítmico ---
    # 1. Converte para Log10
    log_val = np.log10(current_val)

    # 2. Cria candidatos no espaço log (Esquerda, Centro, Direita)
    log_candidates = np.array([log_val - delta_log, log_val, log_val + delta_log])

    # 3. Converte de volta para escala real
    candidates_val = 10 ** log_candidates

    # Arredonda e remove duplicatas
    # ALTERAÇÃO: Arredondamento dinâmico (mantém 4 algarismos significativos)
    candidates_val = sorted(list(set([float(f"{v:.4g}") for v in candidates_val])))

    new_grid = []
    for val in candidates_val:
        # Proteção contra valores inválidos (xi <= 0)
        if val <= 1e-9: continue
        cfg = {
            'acq_func': acq,
            'xi': val if acq in ['EI', 'PI'] else None,
            'kappa': val if acq == 'LCB' else None,
            'tag': f"{acq}_Refined_{val:.5f}"  # Tag atualizada para precisão
        }
        new_grid.append(cfg)

    return new_grid



# 4. MOTOR PRINCIPAL (AUTO-PILOTO)


if __name__ == '__main__':
    print("--- INICIANDO AUTO META-OTIMIZAÇÃO ---")
    print(f"Diretório ANSYS Fixo: {ANSYS_WORKING_DIR}")

    # Grade Inicial (Coarse)
    current_grid = [
        {'acq_func': 'EI', 'xi': 0.001, 'kappa': None, 'tag': 'EI_0.001'},
        {'acq_func': 'EI', 'xi': 0.01, 'kappa': None, 'tag': 'EI_0.01'},
        {'acq_func': 'EI', 'xi': 0.1, 'kappa': None, 'tag': 'EI_0.1'},

        {'acq_func': 'PI', 'xi': 0.001, 'kappa': None, 'tag': 'PI_0.001'},
        {'acq_func': 'PI', 'xi': 0.01, 'kappa': None, 'tag': 'PI_0.01'},
        {'acq_func': 'PI', 'xi': 0.1, 'kappa': None, 'tag': 'PI_0.1'},

        {'acq_func': 'LCB', 'xi': None, 'kappa': 1.0, 'tag': 'LCB_1.0'},
        {'acq_func': 'LCB', 'xi': None, 'kappa': 1.96, 'tag': 'LCB_1.96'},
        {'acq_func': 'LCB', 'xi': None, 'kappa': 5.0, 'tag': 'LCB_5.0'},
    ]

    all_history = []
    iteration = 0
    best_score_global = float('inf')
    best_param_val_global = -1

    while True:
        iteration += 1
        print(f"\n\n=== ITERAÇÃO DE REFINAMENTO {iteration} ===")
        print(f"Testando {len(current_grid)} configurações...")

        # --- Execução da Grade Atual ---
        iter_results = []

        for config in current_grid:
            print(f"\n>>> Config: {config['tag']}")
            fits = []
            times = []

            # Loop Sequencial de Rodadas (Necessário pois usam a mesma pasta)
            for i in range(1, N_REPETICOES + 1):
                print(f"    Run {i}/{N_REPETICOES} ... ", end="")

                queue = Queue()
                p = Process(target=worker_optimization, args=(config, i, queue))
                p.start()

                # Bloqueia até terminar (Sequencial para respeitar pasta única)
                res = queue.get()
                p.join()

                fits.append(res['fitness'])
                times.append(res['time'])

                status = "OK" if res['success'] else "FAIL"
                print(f"{status} (Fit: {res['fitness']:.2e})")

            # Consolida dados
            # Filtra falhas (1e6) se houver sucessos
            valid_fits = [f for f in fits if f < 1e5]
            if not valid_fits: valid_fits = [1e6]

            result_entry = {
                'Iteration': iteration,
                'Tag': config['tag'],
                'Acq': config['acq_func'],
                'Xi': config['xi'],
                'Kappa': config['kappa'],
                'Avg_Fit': np.mean(valid_fits),
                'Std_Fit': np.std(fits),
                'Avg_Time': np.mean(times)
            }
            iter_results.append(result_entry)
            all_history.append(result_entry)

        # --- Análise e Decisão ---
        df_iter = pd.DataFrame(iter_results)
        df_scored = calculate_scores(df_iter)

        best_row = df_scored.iloc[0]
        current_best_score = best_row['Score']

        # Identifica valor do parâmetro vencedor
        param_val = best_row['Xi'] if best_row['Acq'] in ['EI', 'PI'] else best_row['Kappa']

        print("\n--- Resultado da Iteração ---")
        print(df_scored[['Tag', 'Avg_Fit', 'Score']].to_string())
        print(f"Vencedor: {best_row['Tag']} (Score: {current_best_score:.4f})")

        # --- Critérios de Parada ---

        # 1. Melhora insignificante no Score
        score_improv = best_score_global - current_best_score

        # 2. Mudança insignificante no Hiperparâmetro (Convergência)
        param_delta = abs(param_val - best_param_val_global) if best_param_val_global > 0 else float('inf')

        # print(f"Delta Score: {score_improv:.2e} | Delta Param: {param_delta:.2e}")

        # Atualiza globais
        if current_best_score < best_score_global:
            best_score_global = current_best_score
            best_param_val_global = param_val
            best_config_global = best_row

        # Verifica parada (Lógica AND: Param pequeno E Score estagnado)
        if iteration > 1:
            # Avalia condições booleanas
            converged_params = param_delta < MIN_PARAM_DELTA
            stagnated_score = MIN_SCORE_IMPROVEMENT > score_improv > -1e-5

            # Só para se AMBOS forem verdadeiros
            if converged_params and stagnated_score:
                print(f"\n\n>>> PARADA TOTAL: Critérios de convergência atendidos.")
                print(f"   1. Parâmetros estáveis (Delta: {param_delta:.1e} < {MIN_PARAM_DELTA})")
                print(f"   2. Score estagnado (Melhora: {score_improv:.1e} < {MIN_SCORE_IMPROVEMENT})")
                break

            # Logs informativos se apenas um critério for atingido
            elif converged_params:
                print(
                    f"\n[INFO] Parâmetros estabilizaram (Delta {param_delta:.1e}), mas o Score ainda está melhorando. Continuando...")
            elif stagnated_score:
                print(
                    f"\n[INFO] Score estabilizou (Melhora {score_improv:.1e}), mas os Parâmetros ainda estão mudando. Continuando...")

        if iteration >= 9:  # Limite de segurança
            print("\n>>> PARADA: Limite de iterações atingido.")
            break

            # --- Gera Próxima Grade ---
            print("\nGerando grade refinada (Logarítmica)...")

            # Define o passo logarítmico inicial e reduz a cada iteração
            # ALTERAÇÃO AQUI: Começa com 0.5 (Raiz de 10) para cobrir o buraco
            # Reduz suavemente: 0.5 -> 0.33 -> 0.22 -> 0.15
            current_delta_log = 0.5 / (1.5 ** (iteration - 1))

            # Chamada atualizada
            current_grid = generate_refined_grid(best_row, delta_log=current_delta_log)

            # Log de controle
            print(f"   -> Passo Log atual: +/- {current_delta_log:.4f} (Base 10)")

        # Salva Log Parcial
        pd.DataFrame(all_history).to_csv("meta_history_partial.csv", index=False)

    # --- Relatório Final ---
    df_final = pd.DataFrame(all_history)
    df_final.to_csv("meta_history_FINAL.csv", index=False)
    print("\n\n=== AUTO-OTIMIZAÇÃO CONCLUÍDA ===")
    print(f"Melhor Configuração Encontrada: {best_config_global['Tag']}")
    print("Histórico salvo em 'meta_history_FINAL.csv'")