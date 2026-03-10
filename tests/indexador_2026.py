from optimization.parameter import Continuous
from dataclasses import dataclass

@dataclass
class Problema:
    script_filename: str
    noise: float
    parameters: list
    target_params: list
    keys: list
    expected_values: dict
    search_spaces: dict
    key_to_name: dict

@dataclass
class Device:
    base_path: str
    local_path: str


def indexar_problema(problema, full=False):

    script_name = None # script.mac
    noise = None
    key_to_name = {}

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
        key_to_name = {
            'modulo': 'Módulo de Elasticidade',
            'poisson': 'Coeficiente de Poisson',
            'dens': 'Densidade',
            'rigidez1': 'Rigidez do Apoio 1',
            'rigidez2': 'Rigidez do Apoio 2'
        }

    elif problema == 2:
        if full:
            script_name = 'script problema 2 (9 param).mac'
            noise = 0.03
            parameters = [
                Continuous(20e9, 35e9, 'modulo_viga_1'),
                Continuous(20e9, 35e9, 'modulo_viga_2'),
                Continuous(20e9, 35e9, 'modulo_centro'),

                Continuous(0.1, 0.40, 'poisson'),
                Continuous(2400, 2600, 'dens'),

                Continuous(50e6, 50e8, 'rigidez1'),
                Continuous(50e6, 50e8, 'rigidez2'),
                Continuous(50e6, 50e8, 'rigidez3'),
                Continuous(50e6, 50e8, 'rigidez4')
            ]

            # target_params = [32e9, 28e9, 30e9, 0.2, 50e7, 60e7]  # 6
            target_params = [32e9,28e9,30e9,0.2,2500,50e7,40e7,55e7,60e7] # 9
            key_to_name = {
                'modulo_viga_1': 'Módulo de Elasticidade (Viga 1)',
                'modulo_viga_2': 'Módulo de Elasticidade (Viga 2)',
                'modulo_centro': 'Módulo de Elasticidade (Laje)',
                'poisson': 'Coeficiente de Poisson',
                'dens': 'Densidade',
                'rigidez1': 'Rigidez do Apoio 1',
                'rigidez2': 'Rigidez do Apoio 2',
                'rigidez3': 'Rigidez do Apoio 3',
                'rigidez4': 'Rigidez do Apoio 4'
            }
        else:
            script_name = 'script problema 2 (6 param).mac'
            noise = 0.03
            parameters = [
                Continuous(20e9, 35e9, 'modulo_viga_1'),
                Continuous(20e9, 35e9, 'modulo_viga_2'),
                Continuous(20e9, 35e9, 'modulo_centro'),

                Continuous(0.1, 0.40, 'poisson'),
                # Continuous(2400, 2600, 'dens'),

                Continuous(50e6, 50e8, 'rigidez1'),
                # Continuous(50e6, 50e8, 'rigidez2'),
                # Continuous(50e6, 50e8, 'rigidez3'),
                Continuous(50e6, 50e8, 'rigidez4')
            ]

            target_params = [32e9,28e9,30e9,0.2,50e7,60e7] # 6
            # target_params = [32e9,28e9,30e9,0.2,2500,50e7,40e7,55e7,60e7] # 9
            key_to_name = {
                'modulo_viga_1': 'Módulo de Elasticidade (Viga 1)',
                'modulo_viga_2': 'Módulo de Elasticidade (Viga 2)',
                'modulo_centro': 'Módulo de Elasticidade (Laje)',
                'poisson': 'Coeficiente de Poisson',
                # 'dens': 'Densidade',
                'rigidez1': 'Rigidez do Apoio 1',
                # 'rigidez2': 'Rigidez do Apoio 2',
                # 'rigidez3': 'Rigidez do Apoio 3',
                'rigidez4': 'Rigidez do Apoio 4'
            }

    elif problema == 3:
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
        key_to_name = {
            'modulo_banz': 'Módulo de Elasticidade (Grupo 1)',
            'modulo_diag': 'Módulo Elasticidade (Grupo 2)',
            'modulo_contrav': 'Módulo de Elasticidade (Grupo 3)',
            'rigidez1': 'Rigidez do Apoio 1',
            'rigidez2': 'Rigidez do Apoio 2',
            'rigidez3': 'Rigidez do Apoio 3',
            'rigidez4': 'Rigidez do Apoio 4',
            'massa': 'Massa'
        }

    elif problema == 4:
        script_name = "scriptLOP.mac"
        noise = None
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
        target_params = [32.209e9, 15e9, 210e9, 1.1e8, 9.7e7, 0.0417, 8.51e+08, 2.07e+08, 2.15e+07]  # , 4.06e7]
        key_to_name = {
            "modulo_concreto": "Módulo de Elasticidade do Concreto",
            "modulo_madeira": "Módulo de Elasticidade da Madeira",
            "modulo_cordoalhas": "Módulo de Elasticidade do Aço (cordoalhas)",
            "kv": "Rigidez Vertical dos Apoios",
            "kh": "Rigidez Horizontal dos Apoios",
            "h_concreto": "Espessura de Concreto",
            "ey_wood": "Módulo de Elasticidade Transversal da Madeira",
            "GXY": "GXY (madeira)",
            "GYZ": "GYZ (madeira)"
        }

    else:
        raise ValueError(f"Dados não definidos para o problema: {problema}")

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)
    expected_values = dict(zip(keys, target_params))
    search_spaces = {param.key: [param.lower_bound, param.upper_bound] for param in parameters}

    return Problema(script_filename=script_name, noise=noise, parameters=parameters, target_params=target_params, keys=keys, expected_values=expected_values, search_spaces=search_spaces, key_to_name=key_to_name)

def indexar_device(computador):
    # diretórios
    if computador.upper() == "DESKTOP":
        devicepath_base = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\Rodadas"

    elif computador.upper() == "LEST 2":
        devicepath_base = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\Thiago Artur\Documents\Rodadas"

    elif computador.upper() == "LEST 1":
        devicepath_base = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\Thiago\Documents\Rodadas"

    elif computador.upper() == "NOTEBOOK":
        devicepath_base = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\Rodadas"
        devicepath_local = r"C:\Users\thiag\Documentos (Local)\Rodadas"

    else:
        raise ValueError(f"Caminhos não especificados para dispositivo: {computador}")

    return Device(base_path=devicepath_base, local_path=devicepath_local)


## EXEMPLO DE CHAMADA
#
# problema = 1
# computador = "LEST 1"
#
# dados_problema = indexar_problema(problema)
# dados_pc = indexar_device(computador)
#
# print(dados_problema.script_filename)
# print(dados_problema.expected_values)
#
# print(dados_pc.base_path)
