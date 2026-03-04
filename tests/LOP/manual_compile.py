import os

from optimization.parameter import Continuous
from data.compile import compile_convergence_history



ALGOS = ["GA", "PSO", "BO"]
CONJUNTOS = 3

global_log_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 4\teste\log\rodada_20260304_182340"
problema = 4


if problema == 3:

    script_name = "scriptTREL.mac"
    noise = 0.03

    parameters = [
        Continuous(150e9, 250e9, 'modulo_banz'),

        Continuous(150e9, 250e9, 'modulo_diag'),

        Continuous(150e9, 250e9, 'modulo_contrav'),

        Continuous(1e5, 1e7, 'rigidez1'),
        Continuous(1e5, 1e7, 'rigidez2'),
        Continuous(1e5, 1e7, 'rigidez3'),
        Continuous(1e5, 1e7, 'rigidez4'),

        Continuous(400, 800, 'massa')
    ]

    target_params = [205e9, 215e9, 195e9, 8e7, 6.8e7, 7.6e7, 7.2e7, 600]

elif problema == 4:

    script_name = "scriptLOP.mac"

    parameters = [  # analise 10
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

    target_params = [32.209e9, 0.2, 0.6, 15e9, 210e9, 1.1e8, 9.7e7, 1.84e8, 2.07e8, 4.06e7]

else:
    raise ValueError(f"Parâmetros e Gabarito não definidos para o problema: {problema}")

keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)
expected_values = dict(zip(keys, target_params))


for algo in ALGOS:
    algo_dir = os.path.join(global_log_dir, algo)

    for i in range(CONJUNTOS):
        conjunto_nome = f"Conjunto {i + 1}"
        conjunto_log_dir = os.path.join(algo_dir, conjunto_nome)

        compile_convergence_history(algo, expected_values, conjunto_log_dir, conjunto_nome)
