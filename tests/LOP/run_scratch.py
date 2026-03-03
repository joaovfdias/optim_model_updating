from tests.LOP.BO_trel_TestRun import BO_run
from tests.LOP.BO_skopt_trel_TestRun import BO_skopt_run
from tests.LOP.GA_trel_TestRun import GA_run
from tests.LOP.PSO_trel_TestRun import PSO_run
from optimization.parameter import Continuous

import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue


# --- 1. WORKER UNIFICADO (COM QUEUE E TEMPO) ---
def run_algorithm_worker(algo_name, parameters, irun, input_dir, log_dir, queue):
    """
    Roda o algoritmo e coloca os resultados na Queue para o processo principal.
    """
    start_time = time.time()
    try:
        # Executa o algoritmo escolhido
        if algo_name == "PSO":
            best = PSO_run(parameters, irun, input_dir, log_dir)
        elif algo_name == "GA":
            best = GA_run(parameters, irun, input_dir, log_dir)
        elif algo_name == "BO_skopt":
            best = BO_skopt_run(parameters, irun, input_dir, log_dir)
        else:
            raise ValueError("Algoritmo não reconhecido.")

        elapsed = time.time() - start_time

        # Extração robusta dos resultados do 'best' individual
        fit = best.fitness if hasattr(best, 'fitness') else best['fitness']
        params = best.param if hasattr(best, 'param') else best['param']

        # Envia de volta para a main
        queue.put({'success': True, 'fitness': fit, 'time': elapsed, 'params': params})

    except Exception as e:
        print(f"\n[ERRO] Falha no {algo_name} ({irun}): {e}")
        queue.put({'success': False})


# --- 2. FUNÇÃO DE RESUMO E SALVAMENTO (CSV) ---
def summarize_and_save(algo_name, results, expected_params, output_csv):
    """
    Processa os resultados de N rodadas, calcula estatísticas e salva no CSV.
    """
    valid_res = [r for r in results if r['success']]
    if not valid_res:
        print(f">>> Nenhum resultado válido para {algo_name}. Pulando resumo.")
        return

    fits = [r['fitness'] for r in valid_res]
    times = [r['time'] for r in valid_res]

    # Estatísticas Globais
    mean_fit = np.mean(fits)
    std_fit = np.std(fits)
    cv_fit = std_fit / (abs(mean_fit) + 1e-12)
    mean_time = np.mean(times)

    # Cria dicionário de linha para o DataFrame
    row_data = {
        'Algoritmo': algo_name,
        'Rodadas_Validas': len(valid_res),
        'Media_Fit': mean_fit,
        'CV_Fit': cv_fit,
        'Media_Tempo_s': mean_time
    }

    # Estatísticas e Erro por Parâmetro
    for p_name, expected_val in expected_params.items():
        # Coleta o valor encontrado em todas as rodadas
        p_vals = [r['params'][p_name] for r in valid_res]

        mean_p = np.mean(p_vals)
        # Erro Relativo Percentual: |(calculado - esperado) / esperado| * 100
        error_percent = (abs(mean_p - expected_val) / abs(expected_val)) * 100

        row_data[f'{p_name}_Media'] = mean_p
        row_data[f'{p_name}_Erro_%'] = error_percent

    # Converte para Pandas e Salva
    df = pd.DataFrame([row_data])
    file_exists = os.path.isfile(output_csv)

    # Salva no formato PT-BR para abrir bonito no Excel (;) e (,)
    df.to_csv(output_csv, mode='a', index=False, sep=';', decimal=',', header=not file_exists)
    print(f"\n>>> Resumo do {algo_name} calculado e salvo com sucesso em {os.path.basename(output_csv)}")


# --- 3. MAIN (ORQUESTRADOR) ---
if __name__ == '__main__':

    base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 10"
    log_dir = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    csv_resultado_path = os.path.join(log_dir, "resultado_rodadas_automaticas.csv")

    # Definição do Espaço de Busca
    parameters = [
        Continuous(20e9, 35e9, 'modulo_concreto'),
        Continuous(0.1, 0.49, 'poisson_concreto'),
        Continuous(0.02, 0.06, 'h_concreto'),
        Continuous(10e9, 20e9, 'modulo_madeira'),
        Continuous(150e9, 250e9, 'modulo_cordoalhas'),
        Continuous(1e7, 1e9, 'kv'),
        Continuous(1e7, 1e9, 'kh'),
        Continuous(1e6, 1e9, 'GXY'),
        Continuous(1e6, 1e9, 'GYZ'),
        Continuous(1e6, 1e9, 'GXZ')
    ]

    # --- VALORES ESPERADOS (GABARITO) PARA O CÁLCULO DO ERRO ---
    # !! ATENÇÃO: Substitua esses valores pelos verdadeiros do seu problema !!
    expected_values = {
        'modulo_concreto': 25e9,
        'poisson_concreto': 0.20,
        'h_concreto': 0.05,
        'modulo_madeira': 15e9,
        'modulo_cordoalhas': 200e9,
        'kv': 5e8,
        'kh': 5e8,
        'GXY': 5e7,
        'GYZ': 5e7,
        'GXZ': 5e7
    }

    num_runs = 4
    algoritmos = ["PSO", "GA", "BO_skopt"]  # Lista dos algoritmos a rodar

    print(f"{'=' * 50}\nINICIANDO AVALIAÇÃO DE ALGORITMOS\n{'=' * 50}")

    # Loop principal (Algoritmo por Algoritmo)
    for algo in algoritmos:
        print(f"\n--- Iniciando {num_runs} rodadas do algoritmo: {algo} ---")

        algo_results = []

        for irun in range(1, num_runs + 1):
            print(f"  > Executando rodada {irun}/{num_runs}...")

            q = Queue()
            run_name = f"run{irun}"

            p = Process(target=run_algorithm_worker, args=(algo, parameters, run_name, base_dir, log_dir, q))
            p.start()

            # Aguarda a conclusão e coleta resultado
            res = q.get()
            p.join()

            algo_results.append(res)

            if res['success']:
                print(f"    Rodada {irun} concluída. Fit: {res['fitness']:.4e} | Tempo: {res['time']:.2f}s")

        # Após as 4 rodadas do algoritmo atual, calcula e salva o resumo
        summarize_and_save(algo, algo_results, expected_values, csv_resultado_path)

    print(f"\n{'=' * 50}\nTODAS AS AVALIAÇÕES CONCLUÍDAS!\nConsulte o arquivo: {csv_resultado_path}")