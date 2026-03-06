import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Queue

from BO_LOP_run import BO_run
from GA_LOP_run import GA_run
from PSO_LOP_run import PSO_run

from optimization.parameter import Continuous
from data.compile import compile_convergence_history


# --- 1. WORKER UNIFICADO ---
def run_algorithm_worker(algo_name, irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise, hp_kwargs, queue):
    start_time = time.time()
    try:
        if algo_name == "PSO":
            # Repare no "best, evals ="
            best, iters = PSO_run(irun=irun, parameters=parameters, base_dir=base_dir, local_dir=local_dir,
                                  log_dir=log_dir,
                                  base_script_filename=base_script_filename, noise=noise, **hp_kwargs)
        elif algo_name == "GA":
            best, iters = GA_run(irun=irun, parameters=parameters, base_dir=base_dir, local_dir=local_dir,
                                 log_dir=log_dir,
                                 base_script_filename=base_script_filename, noise=noise, **hp_kwargs)
        elif algo_name == "BO":
            best, iters = BO_run(irun=irun, parameters=parameters, base_dir=base_dir, local_dir=local_dir,
                                 log_dir=log_dir,
                                 base_script_filename=base_script_filename, noise=noise, **hp_kwargs)
        else:
            raise ValueError("Algoritmo não reconhecido.")

        elapsed = time.time() - start_time
        fit = best.fitness if hasattr(best, 'fitness') else best['fitness']
        params = best.param if hasattr(best, 'param') else best['param']

        # Converte a lista em um dicionário {'nome_do_parametro': valor}
        # Isso garante que a variável p_name funcione perfeitamente depois
        params_dict = {p.key: valor for p, valor in zip(parameters, params)}

        queue.put({'success': True, 'fitness': fit, 'time': elapsed, 'params': params_dict, 'iters': iters})

    except Exception as e:
        print(f"\n[ERRO WORKER] Falha no {algo_name} ({irun}): {e}")
        queue.put({'success': False})


# --- 2. RESUMO GERAL DOS RESULTADOS FINAIS ---
def summarize_and_save(algo_name, conjunto_nome, results, expected_params, output_csv):
    valid_res = [r for r in results if r['success']]
    if not valid_res:
        return

    fits = [r['fitness'] for r in valid_res]
    times = [r['time'] for r in valid_res]

    # ADICIONADO: Extrai as iterações realmente executadas
    iters = [r['iters'] for r in valid_res]

    mean_fit, std_fit = np.mean(fits), np.std(fits)
    cv_fit = std_fit / (abs(mean_fit) + 1e-12)

    row_data = {
        'Algoritmo': algo_name,
        'Hiperparâmetros': conjunto_nome,
        'Rodadas_Validas': len(valid_res),
        'Media_Iteracoes': np.mean(iters),
        'Desvio_Iteracoes': np.std(iters),
        'Media_Fit': mean_fit,
        'CV_Fit': cv_fit,
        'Media_Tempo_s': np.mean(times)
    }

    for p_name, expected_val in expected_params.items():
        p_vals = [r['params'][p_name] for r in valid_res]
        mean_p = np.mean(p_vals)
        row_data[f'{p_name}_Media'] = mean_p
        row_data[f'{p_name}_Erro_%'] = (abs(mean_p - expected_val) / abs(expected_val))

    df = pd.DataFrame([row_data])
    file_exists = os.path.isfile(output_csv)
    df.to_csv(output_csv, mode='a', index=False, sep=';', decimal='.', header=not file_exists)

    print(f"\n[RESUMO] Atualizado com {conjunto_nome} do algoritmo {algo_name} em: {output_csv}")


# --- 3. COMPILADOR DE CONVERGÊNCIA (NOVIDADE) ---
    # foi movido para data.compile

# --- 4. ORQUESTRADOR ---
if __name__ == '__main__':

    problema = 1
    teste = False
    apenas_conjunto_medio = True
    num_runs = 4 if not teste else 2

    computador = "LEST 1"

    # diretórios
    if computador == "LEST 2":
        devicepath_base = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\Thiago Artur\Documents\Rodadas"

    if computador == "LEST 1":
        devicepath_base = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\Thiago\Documents\Rodadas"

    base_dir = os.path.join(devicepath_base, f"Problema {problema}")
    local_dir = os.path.join(devicepath_local, f"Problema {problema}") # copia ModBase.db pro diretório local
    if teste: local_dir = os.path.join(local_dir, "teste")
    os.makedirs(local_dir, exist_ok=True)

    initimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_log_dir = os.path.join(base_dir, "log", f"rodada_{initimestamp}") if not teste else os.path.join(base_dir, "teste", "log", f"rodada_{initimestamp}")
    os.makedirs(global_log_dir, exist_ok=True)

    csv_resultado_path = os.path.join(global_log_dir, f"Resumo Global - Problema {problema}.csv")

    # definição dos parâmetros do problema
    script_name = None
    noise = None

    if problema == 1:

        script_name = 'scriptVIGA.mac'
        noise = None

        parameters = [
            Continuous(20e9, 30e9, 'modulo'),
            Continuous(0.1, 0.49, 'poisson'),
            Continuous(2400, 2600, 'dens'),
            Continuous(10e6, 10e8, 'rigidez1'),
            Continuous(10e6, 10e8, 'rigidez2')
        ]

        target_params = [23e9, 0.2, 2500, 1e7, 1.5e7]

    if problema == 3:

        script_name = "scriptTREL.mac"
        noise = 0.03

        parameters = [
            Continuous(180e9, 220e9, 'modulo_banz'),

            Continuous(180e9, 220e9, 'modulo_diag'),

            Continuous(180e9, 220e9, 'modulo_contrav'),

            Continuous(1e7, 1e8, 'rigidez1'),
            Continuous(1e7, 1e8, 'rigidez2'),
            Continuous(1e7, 1e8, 'rigidez3'),
            Continuous(1e7, 1e8, 'rigidez4'),

            Continuous(400, 800, 'massa')
        ]

        target_params = [205e9, 215e9, 195e9, 8e7, 6.8e7, 7.6e7, 7.2e7, 600]

    elif problema == 4:

        script_name = "scriptLOP.mac"

        parameters = [  # GXZ fixado em 1e8, Ey incluido
            Continuous(29.2e9, 33e9, 'modulo_concreto'),
            Continuous(12e9, 18e9, 'modulo_madeira'),
            Continuous(200e9, 220e9, 'modulo_cordoalhas'),

            Continuous(5e7, 5e8, 'kv'),
            Continuous(5e7, 5e8, 'kh'),

            Continuous(0.04, 0.06, 'h_concreto'),

            Continuous(1e6, 1e9, 'ey_wood'),  # esperado 10 a 500 MPa
            Continuous(1e7, 1e9, 'GXY'),  # 600-900 MPa  6e8
            Continuous(1e6, 1e8, 'GYZ')  # 50-150 MPa  5e7
        ]

        target_params = [32.209e9, 0.0417, 15e9, 210e9, 1.1e8, 9.7e7, 8.51e+08, 2.07e+08, 2.15e+07] #, 4.06e7]

    else:
        raise ValueError(f"Parâmetros e Gabarito não definidos para o problema: {problema}")

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)
    expected_values = dict(zip(keys, target_params))


    # SEUS DOIS (agora três) CONJUNTOS DE HIPERPARÂMETROS
    # conjunto 1: explorador (P2 COM 9 PARAM)
    # conjunto 2: médio
    # conjunto 3: intensificador (P2 COM 6 PARAM)

    if teste: # conjunto de teste (rodadas rapidinhas)
        configs_algoritmos = {
            "PSO": [
                {"population_size": len(parameters), "iterations": 4, "w": 0.73, "w_rate": 0.957, "c1": 1.90, "c2": 1.32,
                 "init_vel_ratio": 0.06},
                {"population_size": len(parameters), "iterations": 4, "w": 0.6, "w_rate": 0.99, "c1": 2.05, "c2": 2.05,
                 "init_vel_ratio": 0.20},
                {"population_size": len(parameters), "iterations": 4, "w": 1.13, "w_rate": 0.964, "c1": 1.15, "c2": 1.43,
                 "init_vel_ratio": 0.18}
            ],
            "BO": [
                {"initial_points": len(parameters), "evaluations": 10, "acq_func": 'PI', "xi": 0.1},
                {"initial_points": len(parameters), "evaluations": 10, "acq_func": 'gp_hedge'},
                {"initial_points": len(parameters), "evaluations": 10, "acq_func": 'EI', "xi": 0.003162}
            ],
            "GA": [
                {"population_size": len(parameters), "generations": 4, "elitism_rate": 0.12, "crossover_rate": 0.87,
                 "mutation_strength": 0.185},
                {"population_size": len(parameters), "generations": 4, "elitism_rate": 0.10, "crossover_rate": 0.60,
                 "mutation_strength": 0.10},
                {"population_size": len(parameters), "generations": 4, "elitism_rate": 0.10, "crossover_rate": 0.75,
                 "mutation_strength": 0.25}
            ]
        }

    else:
        configs_algoritmos = {
            "PSO": [
                {"population_size": None, "iterations": None, "w": 0.73, "w_rate": 0.957, "c1": 1.90, "c2": 1.32,
                 "init_vel_ratio": 0.06},
                {"population_size": None, "iterations": None, "w": 0.6, "w_rate": 0.99, "c1": 2.05, "c2": 2.05,
                 "init_vel_ratio": 0.20},
                {"population_size": None, "iterations": None, "w": 1.13, "w_rate": 0.964, "c1": 1.15, "c2": 1.43,
                 "init_vel_ratio": 0.18}
            ],
            "BO": [
                {"initial_points": None, "evaluations": None, "acq_func": 'PI', "xi": 0.1},
                {"initial_points": None, "evaluations": None, "acq_func": 'gp_hedge'},
                {"initial_points": None, "evaluations": None, "acq_func": 'EI', "xi": 0.003162}
            ],
            "GA": [
                {"population_size": None, "generations": None, "elitism_rate": 0.12, "crossover_rate": 0.87,
                 "mutation_strength": 0.185},
                {"population_size": None, "generations": None, "elitism_rate": 0.10, "crossover_rate": 0.60,
                 "mutation_strength": 0.10},
                {"population_size": None, "generations": None, "elitism_rate": 0.10, "crossover_rate": 0.75,
                 "mutation_strength": 0.25}
            ]
        }

    if apenas_conjunto_medio:
        configs_do_meio = {algoritmo: [configs[1]] for algoritmo, configs in configs_algoritmos.items()}
        configs_algoritmos = configs_do_meio

    print(f"{'=' * 60}\nINICIANDO AVALIAÇÃO DE ALGORITMOS (RODADA {initimestamp})\n{'=' * 60}")

    for algo, conjuntos in configs_algoritmos.items():
        print(f"\n[{algo}] Configurando diretórios...")

        algo_dir = os.path.join(global_log_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

        csv_hp_path = os.path.join(algo_dir, f"hiperparametros_{algo}.csv")
        # Prepara a lista de dicionários adicionando a coluna 'CONJUNTO' no início
        hp_data = []
        for idx, config in enumerate(conjuntos):
            row = {"CONJUNTO": f"Conjunto {idx + 1}"}
            row.update(config)  # Adiciona os hiperparâmetros (w, c1, c2, etc)
            hp_data.append(row)
        # Converte para DataFrame e salva
        df_hp = pd.DataFrame(hp_data)
        df_hp.to_csv(csv_hp_path, sep=';', decimal='.', index=False)

        for idx, config in enumerate(conjuntos):
            conjunto_nome = f"Conjunto {idx + 1}"
            conjunto_log_dir = os.path.join(algo_dir, conjunto_nome)
            os.makedirs(conjunto_log_dir, exist_ok=True)

            print(f"\n  --- {algo} | {conjunto_nome} ---")

            algo_results = []
            max_attempts = 3
            for irun in range(1, num_runs + 1):
                print(f"\n    > Executando repetição {irun}/{num_runs}...")

                attempt = 1
                success_run = False
                res = None

                while attempt <= max_attempts:

                    q = Queue()

                    p = Process(target=run_algorithm_worker,
                                args=(algo, irun, parameters, base_dir, local_dir, conjunto_log_dir, script_name, noise,
                                      config, q))
                    p.start()
                    res = q.get()
                    p.join()

                    if res['success']:
                        success_run = True
                        print(f"\n[OK] Fit: {res['fitness']:.4e} | Tempo: {res['time']:.2f}s")
                        break  # Deu certo! Sai do loop while (de tentativas) e vai pra próxima rodada.
                    else:
                        print(f"\n      [FALHA] A rodada {irun} falhou (Tentativa {attempt}/{max_attempts}).")
                        attempt += 1

                        if attempt <= max_attempts:
                            print(f"      [RETRY] Aguardando 4 segundos para limpar memória e tentar novamente...")
                            time.sleep(4)  # Tempo vital para o Windows/ANSYS soltar arquivos .lock

                # Após o while: adiciona o resultado (seja o sucesso final ou a falha após esgotar tentativas)
                algo_results.append(res)

                if not success_run:
                    print(
                        f"\n      [ERRO CRÍTICO] Rodada {irun} abortada definitivamente após {max_attempts} tentativas.")

            summarize_and_save(algo, conjunto_nome, algo_results, expected_values, csv_resultado_path)

            # CHAMA A NOVA FUNÇÃO DE COMPILAÇÃO APÓS AS 4 RODADAS
            compile_convergence_history(algo, expected_values, conjunto_log_dir, conjunto_nome)

    print(f"\n{'=' * 60}\n\nAVALIAÇÕES CONCLUÍDAS!\nRegistros em: {global_log_dir}")