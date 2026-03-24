from TuRBO_run import TuRBO_run
from tests.indexador_2026 import *

from multiprocessing import Process
import os


def run_TuRBO(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, initial_points=None, evaluations=None, batch_size=4, acqf="ts"):
    TuRBO_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise, initial_points, evaluations, batch_size, acqf)

if __name__ == '__main__':

    Problema = 3
    Compiuter = "LEST 2"
    runs = 10

    pb = indexar_problema(Problema)
    pc = indexar_device(Compiuter)

    base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
    local_dir = os.path.join(pc.local_path, f"Problema {Problema}")

    script_name = pb.script_filename
    noise = pb.noise
    parameters = pb.parameters

    for irun in range(1, runs+1):
        print(f"\nRunning TuRBO ({irun}/{runs}). . .")

        p = Process(target=run_TuRBO, args=(irun, parameters, base_dir, local_dir, None, script_name, noise))
        p.start()
        p.join()

import time

from turbo_metaopt import MetaTuRBO


def main():

    # ==========================================
    # CONFIGURAÇÃO DOS PROBLEMAS
    # ==========================================
    # 🔧 ALTERE AQUI RAPIDAMENTE
    problem_ids = [1, 2, 3, 4]   # usa indexar_problema()

    # ==========================================
    # CONFIGURAÇÃO DA META-OTIMIZAÇÃO
    # ==========================================
    meta = MetaTuRBO(
        problem_ids=problem_ids,
        evaluations=100,      # budget reduzido do TuRBO
        batch_size=4,
        n_runs=5,             # 🔴 5 rodadas por problema
        normalize=True
    )

    # ==========================================
    # EXECUÇÃO
    # ==========================================
    print("\n[Meta] Iniciando meta-otimização do TuRBO...")
    t0 = time.time()

    result_meta, best_params = meta.run(
        n_calls=20,           # número de avaliações do gp_minimize
        n_initial_points=5,
        random_state=42
    )

    t1 = time.time()

    # ==========================================
    # RESULTADOS
    # ==========================================
    print("\n==========================================")
    print("META-OTIMIZAÇÃO FINALIZADA")
    print("==========================================")

    print(f"\nTempo total: {t1 - t0:.2f} s")

    print("\nMelhores hiperparâmetros encontrados:")
    print(f"length            = {best_params['length']:.4f}")
    print(f"success_tolerance = {best_params['success_tol']}")
    print(f"failure_tolerance = {best_params['failure_tol']}")
    print(f"n_init            = {best_params['n_init']}")

    print("\nMelhor valor da função objetivo:")
    print(result_meta.fun)

    print("\nHistórico das avaliações (opcional):")
    for i, val in enumerate(result_meta.func_vals):
        print(f"Iter {i+1:02d}: {val:.6f}")


if __name__ == "__main__":
    main()