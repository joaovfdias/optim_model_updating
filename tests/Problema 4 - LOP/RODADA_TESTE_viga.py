from RODADA_TESTE_main import avaliar_rodada
from optimization.parameter import Continuous

base_dir = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 1"
local_dir = r"C:\Users\thiag\Documentos (Local)\Problema 1 (2026)"

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

avaliar_rodada(parameters, target_params, base_dir, local_dir, script_name, noise)