import os
from datetime import datetime

from tests.indexador_2026 import indexar_device
from turbo_metaopt import MetaTuRBO


def executar_metaopt(nome_etapa, lista_problemas, device, log_dir=None):
    """
    Função auxiliar para instanciar e rodar a meta-otimização,
    evitando repetição de código.
    """
    print("\n" + "=" * 70)
    print(f" INICIANDO ETAPA: {nome_etapa}")
    print(f" Problemas Alvo: {lista_problemas}")
    print("=" * 70)

    # Configurações de orçamento (ajuste conforme necessário)
    EVALS_TURBO = 400
    BATCH_SIZE = 4
    NUM_RODADAS = 5

    # Se for mais de 1 problema, é recomendável ligar a normalização
    # para que um problema com fitness escala 1000 não ofusque um de escala 0.1
    usar_normalizacao = len(lista_problemas) > 1

    meta_optimizer = MetaTuRBO(
        problem_ids=lista_problemas,
        device=device,
        evaluations=EVALS_TURBO,
        batch_size=BATCH_SIZE,
        n_runs=NUM_RODADAS,
        normalize=usar_normalizacao,
        seeds=(42, 100, 333)
    )

    meta_optimizer.log_dir = log_dir

    start_time = datetime.now()

    # Executa o gp_minimize do skopt
    result, best_params = meta_optimizer.run(
        n_calls=80,  # Orçamento do BO
        n_initial_points=8,  # Exploração inicial (Random/Sobol)
        random_state=None
    )

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60.0

    print("\n" + "-" * 70)
    print(f" RESULTADO: {nome_etapa}")
    print(f" Tempo de execução: {total_time:.2f} minutos")
    print(f" Melhor Score (J): {result.fun:.5f}")
    print(" Configuração Ótima (TuRBO):")
    for param_name, value in best_params.items():
        print(f"   -> {param_name}: {value}")
    print("-" * 70 + "\n")

    return best_params


if __name__ == "__main__":
    # # Garante que as pastas raiz existam
    # os.makedirs("log", exist_ok=True)
    # os.makedirs("output", exist_ok=True)

    DEVICE_LOCAL = "LEST 1"  # Ou "notebook"
    pc = indexar_device(DEVICE_LOCAL)

    # Lista de todos os problemas que você quer meta-otimizar
    TODOS_OS_PROBLEMAS = [4]

    melhores_configs_individuais = {}

    # =====================================================================
    # FASE 1: META-OTIMIZAÇÃO INDIVIDUAL (Especialista para cada topologia)
    # =====================================================================
    for prob_id in TODOS_OS_PROBLEMAS:
        nome_etapa = f"Calibração Específica - Problema {prob_id}"

        # Passa a lista com apenas 1 elemento
        best_params = executar_metaopt(nome_etapa, [prob_id], DEVICE_LOCAL)
        melhores_configs_individuais[prob_id] = best_params

    # =====================================================================
    # FASE 2: META-OTIMIZAÇÃO GLOBAL (Generalista) - OPCIONAL
    # =====================================================================
    RODAR_GLOBAL = False
    GLOBAL_LOG = os.path.join(pc.base_dir, "metaopt", "log")

    if RODAR_GLOBAL and len(TODOS_OS_PROBLEMAS) > 1:
        nome_etapa = "Calibração Global (Todos os Problemas)"

        # Passa a lista completa. O seu código vai calcular a média de todos!
        best_params_global = executar_metaopt(nome_etapa, TODOS_OS_PROBLEMAS, DEVICE_LOCAL, log_dir=GLOBAL_LOG)

    # =====================================================================
    # RESUMO FINAL
    # =====================================================================
    print("\n" + "=" * 70)
    print(" RESUMO FINAL DAS CONFIGURAÇÕES IDEAIS DO TuRBO")
    print("=" * 70)
    for prob_id, config in melhores_configs_individuais.items():
        print(f"Problema {prob_id}: {config}")

    if RODAR_GLOBAL and len(TODOS_OS_PROBLEMAS) > 1:
        print(f"\nGlobal (Média): {best_params_global}")
    print("=" * 70)