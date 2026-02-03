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
ANSYS_EXE_PATH = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
# Pasta FIXA onde o ModBase.db já deve estar (sem subpastas)

# LEST
BASE_DIR = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros" # LEST
ANSYS_WORKING_DIR = os.path.join(BASE_DIR, 'ANSYS')

# CASA
# BASE_DIR = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros" # CASA
# ANSYS_WORKING_DIR = os.path.join(BASE_DIR, 'ANSYS','HOME')

INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

# Configuração da Otimização
N_REPETICOES = 3  # Rodadas por configuração para média estatística
N_INITIAL_POINTS = 45  # Pontos Iniciais
N_EVALUATIONS = 195  # Total (45 iniciais + 105 iterações do BO)

# Critérios de Parada do Refinamento
MIN_PARAM_DELTA = 1e-3  # Se a mudança no Xi/Kappa for menor que isso, para.
MIN_SCORE_IMPROVEMENT = 1e-4  # Se o score não melhorar isso, para.

# Pesos para o Ranking (Score)
W_FITNESS = 0.6
W_TIME = 0.2
W_STD = 0.2



# 2. WORKER (Execução Isolada)

def worker_optimization(config, run_id, result_queue):
    """Executa uma otimização completa em processo isolado com Ruído nos dados de referência."""
    try:
        # A. Definição dos Parâmetros (Laje)
        parameters = [
            Continuous(20e9, 35e9, 'modulo_viga_1'),
            Continuous(20e9, 35e9, 'modulo_viga_2'),
            Continuous(20e9, 35e9, 'modulo_centro'),
            # Continuous(20e9, 35e9, 'modulo_borda_1'),
            # Continuous(20e9, 35e9, 'modulo_borda_2'),

            Continuous(0.1, 0.40, 'poisson'),
            Continuous(2400, 2600, 'dens'),

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

        # --- APLICAÇÃO DE RUÍDO NOS DADOS DE REFERÊNCIA ---
        # Simula incerteza experimental diferente para cada rodada
        # Nível de Ruído (Sigma): 1% (0.01) ou 3% (0.03) são valores comuns
        NOISE_LEVEL = 0.03

        # Semente aleatória única para este processo (garante variação entre workers)
        np.random.seed(int(time.time()) + run_id)

        # 1. Ruído nas Frequências (Multiplicativo)
        # freq_new = freq_old * (1 + N(0, sigma))
        freq_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_freq.shape)
        ansys.base_freq = ansys.base_freq * (1 + freq_noise)

        # 2. Ruído nos Modos (Multiplicativo)
        # Afeta a amplitude de cada ponto do modo
        mode_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_modes.shape)
        ansys.base_modes = ansys.base_modes * (1 + mode_noise)

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
        log_dir = os.path.join(BASE_DIR, "meta-opt", 'BO_auto_meta_logs')
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

        # # Limpeza Final (Force Kill ANSYS para liberar .lock da pasta)
        # try:
        #     ansys.mapdl.exit()
        # except:
        #     pass

    except Exception as e:
        print(f"[ERRO WORKER: Run {run_id}] {e}")
        result_queue.put({'fitness': None, 'time': 0, 'success': False})



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


# ... (Imports e funções anteriores permanecem iguais) ...

if __name__ == '__main__':
    print("--- INICIANDO AUTO META-OTIMIZAÇÃO (POR FAMÍLIA) ---")
    print(f"Diretório ANSYS Fixo: {ANSYS_WORKING_DIR}")

    # 1. Definimos as famílias e seus pontos de partida (Coarse)
    # Cada bloco será otimizado isoladamente até convergir
    optimization_families = {
        'EI': [
            {'acq_func': 'EI', 'xi': 0.001, 'kappa': None, 'tag': 'EI_Start_0.001'},
            {'acq_func': 'EI', 'xi': 0.01, 'kappa': None, 'tag': 'EI_Start_0.01'},
            {'acq_func': 'EI', 'xi': 0.1, 'kappa': None, 'tag': 'EI_Start_0.1'},
        ],
        'PI': [
            {'acq_func': 'PI', 'xi': 0.001, 'kappa': None, 'tag': 'PI_Start_0.001'},
            {'acq_func': 'PI', 'xi': 0.01, 'kappa': None, 'tag': 'PI_Start_0.01'},
            {'acq_func': 'PI', 'xi': 0.1, 'kappa': None, 'tag': 'PI_0.1'},
        ],
        'LCB': [
            {'acq_func': 'LCB', 'xi': None, 'kappa': 1.0, 'tag': 'LCB_Start_1.0'},
            {'acq_func': 'LCB', 'xi': None, 'kappa': 1.96, 'tag': 'LCB_Start_1.96'},
            {'acq_func': 'LCB', 'xi': None, 'kappa': 5.0, 'tag': 'LCB_Start_5.0'},
        ]
    }

    global_history = []  # Guardará o histórico de TODAS as famílias

    metatimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- LOOP EXTERNO: Itera sobre cada Família (EI, depois PI, depois LCB) ---
    for fam_name, initial_grid in optimization_families.items():
        print(f"\n\n{'-' * 60}")
        print(f"INICIANDO OTIMIZAÇÃO DA FAMÍLIA: {fam_name}")
        print(f"{'-' * 60}")

        current_grid = initial_grid

        # Variáveis de controle locais desta família
        iteration = 0
        best_score_local = float('inf')
        best_param_val_local = -1

        # --- LOOP DE REFINAMENTO (Zoom) ESPECÍFICO DA FAMÍLIA ---
        while True:
            iteration += 1
            print(f"\n--- {fam_name} | Iteração {iteration} ---")

            iter_results = []

            # Executa a grade atual
            for config in current_grid:
                print(f"\n>>> Config: {config['tag']}")
                fits = []
                times = []

                # parâmetros para repetição de tentativa falha
                run_idx = 1
                max_consecutive_fails = 5 # tentativas até dar pass
                fails_count = 0

                # Loop de Repetições Estatísticas -> alterado para While para comportar repetição de falhas
                while run_idx <= N_REPETICOES:
                    print(f"    Run {run_idx}/{N_REPETICOES} ... ", end="")
                    queue = Queue()
                    # lançamento do processo
                    p = Process(target=worker_optimization, args=(config, run_idx, queue))
                    p.start()
                    # bloqueia no aguardo de resposta:
                    res = queue.get()
                    p.join()

                    if res['success']:
                        # SUCESSO: Registra e avança
                        fits.append(res['fitness'])
                        times.append(res['time'])
                        print(f"OK (Fit: {res['fitness']:.2e})")

                        run_idx += 1  # Avança para a próxima rodada
                        fails_count = 0  # Reseta contador de falhas
                    else:
                        # FALHA: Não avança run_idx, tenta novamente
                        fails_count += 1
                        print(f"FALHA (Tentativa {fails_count}/{max_consecutive_fails}). Tentando novamente...")

                        time.sleep(1.5)  # Respiro para o Windows liberar arquivos .lock

                        # Segurança: Se falhar 5x seguidas na MESMA rodada, aborta a config inteira
                        if fails_count >= max_consecutive_fails:
                            print(
                                f"\n[ERRO CRÍTICO] Configuração {config['tag']} falhou {max_consecutive_fails}x seguidas. Abortando config.")
                            # Preenche com penalidade para não quebrar o log e sai do while
                            fits.append(1e6)
                            times.append(0)
                            break

                # Consolida Dados
                valid_fits = [f for f in fits if f < 1e5]
                if not valid_fits: valid_fits = [1e6]

                entry = {
                    'Family': fam_name,  # Identificador novo
                    'Iteration': iteration,
                    'Tag': config['tag'],
                    'Acq': config['acq_func'],
                    'Xi': config['xi'],
                    'Kappa': config['kappa'],
                    'Avg_Fit': np.mean(valid_fits),
                    'Std_Fit': np.std(fits),
                    'Avg_Time': np.mean(times)
                }
                iter_results.append(entry)
                # global_history.append(entry)

            # Análise da Iteração Local
            df_iter = pd.DataFrame(iter_results)
            df_scored = calculate_scores(df_iter)  # calcula e adiciona a coluna 'Score'

            # Adiciona os resultados JÁ COM SCORE ao histórico global
            # .to_dict('records') converte o DataFrame pontuado em lista de dicionários
            global_history.extend(df_scored.to_dict('records'))

            best_row = df_scored.iloc[0]
            current_best_score = best_row['Score']

            # Parâmetro vencedor local
            param_val = best_row['Xi'] if fam_name in ['EI', 'PI'] else best_row['Kappa']

            print(f"\nVencedor {fam_name} Rodada {iteration}: {best_row['Tag']} (Score: {current_best_score:.4f})")

            # --- Critérios de Parada Locais ---
            score_improv = best_score_local - current_best_score
            param_delta = abs(param_val - best_param_val_local) if best_param_val_local > 0 else float('inf')

            # Atualiza o melhor local
            if current_best_score < best_score_local:
                best_score_local = current_best_score
                best_param_val_local = param_val

            # Verifica Parada (AND)
            if iteration > 1:
                converged = param_delta < MIN_PARAM_DELTA
                stagnated = MIN_SCORE_IMPROVEMENT > score_improv > -1e-5

                if converged and stagnated:
                    print(f">>> {fam_name} CONVERGIU. Indo para próxima família.")
                    break

            if iteration >= 9:  # O tal limite de segurança do zoom
                print(f">>> {fam_name} atingiu limite de iterações.")
                break

            # Gera Próxima Grade para esta família
            delta_log = 0.5 / (1.5 ** (iteration - 1))
            current_grid = generate_refined_grid(best_row, delta_log=delta_log)

            # Salva parcial seguro
            pd.DataFrame(global_history).to_csv(os.path.join(BASE_DIR, "meta-opt", f"meta_history_partial_{metatimestamp}.csv"), index=False)

    Ansys.kill_ansys_process() # matar processo ao final das rodadas

    # --- Relatório Final Unificado ---
    df_final = pd.DataFrame(global_history)
    df_final.to_csv(os.path.join(BASE_DIR, f"meta_history_FINAL_BY_FAMILY{metatimestamp}.csv"), index=False)
    print("\n=== TODAS AS FAMÍLIAS PROCESSADAS ===")
    print("Consulte 'meta_history_FINAL_BY_FAMILY.csv' para comparar os melhores de cada grupo.")