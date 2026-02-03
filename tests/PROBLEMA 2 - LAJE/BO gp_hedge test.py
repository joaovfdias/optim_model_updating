import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Queue

from optimization.parameter import Continuous
from external.parser import Ansys
from external.special_functions import SpecialFun
from optimization.bo_optimizer.bayesian_from_skopt import BO


# 1. CONFIGURAÇÕES ESPECÍFICAS


# Defina aqui os valores fixos vencedores
MELHOR_XI = 0.01  # Exemplo: Substitua pelo seu valor
MELHOR_KAPPA = 1.96  # Exemplo: Substitua pelo seu valor

# Configuração da Rodada
TAG_RODADA = f"GP_Hedge_Final_Xi{MELHOR_XI}_K{MELHOR_KAPPA}"
N_REPETICOES = 5
N_EVALUATIONS = 150  # Aumentar se quiser mais exploração, como conversamos

# Caminhos (Mantidos iguais)
ANSYS_EXE_PATH = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
BASE_DIR = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
ANSYS_WORKING_DIR = os.path.join(BASE_DIR, 'ANSYS')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')


# 2. WORKER

def worker_optimization(run_id, xi_val, kappa_val, result_queue):
    try:
        # A. Parametros
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

        # B. ANSYS
        ansys_output_dir = OUTPUT_DIR
        ansys = Ansys(ANSYS_EXE_PATH, ANSYS_WORKING_DIR, INPUT_DIR, "script problema 2.mac",
                      "target_freq.txt", "target_modes.txt", ansys_output_dir)
        ansys.set_output_filenames("out_freq.txt", "out_modes.txt")
        ansys.max_attempts = 6

        # C. Ruído (consistência com as rodadas anteriores)
        NOISE_LEVEL = 0.03
        np.random.seed(int(time.time()) + run_id)
        ansys.base_freq = ansys.base_freq * (1 + np.random.normal(0, NOISE_LEVEL, ansys.base_freq.shape))
        ansys.base_modes = ansys.base_modes * (1 + np.random.normal(0, NOISE_LEVEL, ansys.base_modes.shape))

        # D. Função
        def fitness_function(param):
            input_file = ansys.create_input_file(param, keys)
            ansys.run_ansys(input_file, True, True)
            try:
                comp_freq = ansys.read_frequencies()
                comp_modes = ansys.read_modes()
                paired_f, paired_m, mac_err = SpecialFun.pair_modes_mac(comp_freq, comp_modes, ansys.base_modes)
                freq_err = SpecialFun.norm_freq_errors(ansys.base_freq, paired_f)
                return (1.0 * freq_err) + (1.0 * mac_err), {}
            except:
                return 1e6, {}

        # E. Otimizador GP_Hedge
        log_dir = os.path.join(BASE_DIR, 'final_logs')
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        optimizer = BO(fitness_function, parameters, 45)  # 45 Pontos iniciais
        optimizer.set_sampling_method('lhs')
        optimizer.set_log(f"{TAG_RODADA}_R{run_id}", log_dir)

        start_t = time.time()

        # CHAMADA COM GP_HEDGE
        result = optimizer.run(
            evaluations=N_EVALUATIONS,
            acq_func='gp_hedge',  # Força o uso do portfólio
            xi=xi_val,  # Usa seus melhores parâmetros para as funções internas
            kappa=kappa_val,
            status=False, log=True
        )
        elapsed = time.time() - start_t

        result_queue.put({'fitness': result.fun, 'time': elapsed, 'success': True})

    except Exception as e:
        print(f"[ERRO] {e}")
        result_queue.put({'fitness': 1e6, 'time': 0, 'success': False})



# 3. MAIN (Execução Simples)

if __name__ == '__main__':
    print(f"--- INICIANDO RODADA FINAL: {TAG_RODADA} ---")
    print(f"Algoritmo: gp_hedge | Xi: {MELHOR_XI} | Kappa: {MELHOR_KAPPA}")

    fits = []
    times = []

    # Loop de 5 repetições
    for i in range(1, N_REPETICOES + 1):
        print(f"Run {i}/{N_REPETICOES}... ", end="")
        q = Queue()
        p = Process(target=worker_optimization, args=(i, MELHOR_XI, MELHOR_KAPPA, q))
        p.start()
        res = q.get()
        p.join()

        if res['success']:
            fits.append(res['fitness'])
            times.append(res['time'])
            print(f"OK (Fit: {res['fitness']:.4f})")
        else:
            print("FALHA")

    # Cálculos Finais
    avg_fit = np.mean(fits)
    std_fit = np.std(fits)
    avg_time = np.mean(times)

    # Cria DataFrame Único
    df_result = pd.DataFrame([{
        'Tag': TAG_RODADA,
        'Acq': 'gp_hedge',
        'Xi': MELHOR_XI,
        'Kappa': MELHOR_KAPPA,
        'Avg_Fit': avg_fit,
        'Std_Fit': std_fit,
        'Avg_Time': avg_time,
        'Raw_Fits': str(fits)
    }])

    # Salva
    filename = f"Resultado_Final_GPHedge_{datetime.now().strftime('%H%M')}.csv"
    df_result.to_csv(filename, index=False)
    print(f"\nConcluído! Resultado salvo em {filename}")
    print(df_result.to_string())

    try:
        Ansys.kill_ansys_process()
    except:
        pass