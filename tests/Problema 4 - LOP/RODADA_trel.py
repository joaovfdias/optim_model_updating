from RODADA_TESTE_main import avaliar_rodada
from optimization.parameter import Continuous

base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 3"
local_dir = r"C:\Users\Thiago Artur\Documents\Rodadas\Problema 3"

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

avaliar_rodada(parameters, target_params, base_dir, local_dir, script_name, noise)