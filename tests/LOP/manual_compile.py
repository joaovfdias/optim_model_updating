import os

from optimization.parameter import Continuous
from data.compile import compile_convergence_history
from tests.indexador_2026 import indexar_problema



ALGOS = ["GA", "PSO", "BO"]
CONJUNTOS = 3

# pasta raiz da rodada com estrutura de LOP\auto_run.py
global_log_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 4\Resultados"
problema = 4

# definição dos parâmetros do problema
dadosprob = indexar_problema(problema)

script_name = dadosprob.script_filename
noise = dadosprob.noise
parameters = dadosprob.parameters
target_params = dadosprob.target_params
keys = dadosprob.keys
expected_values = dadosprob.expected_values


for algo in ALGOS:
    algo_dir = os.path.join(global_log_dir, algo)

    for i in range(CONJUNTOS):
        conjunto_nome = f"Conjunto {i + 1}_novo"
        conjunto_log_dir = os.path.join(algo_dir, f"Conjunto {i + 1}")

        compile_convergence_history(algo, expected_values, conjunto_log_dir, conjunto_nome)
