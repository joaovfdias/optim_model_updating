from optimization.parameter import Continuous

import os


# 1. DEFINIÇÃO DOS DADOS DO PROBLEMA

# 1.1. LISTA DE PARÂMETROS
    # lista de objetos derivados da classe Parameter
        # para esse caso, valores contínuos com limites de busca e identificador

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

# lista apenas com identificadores dos parâmetros (equivalente ao %key% no script.mac)
keys = [parameter.key for parameter in parameters]


# 1.2. CAMINHOS
    # diretório onde os arquivos necessários se encontram:
        # pasta input
            # script MAPDL
            # dados de referência
        # modelo base (se houver)

# todos os caminhos de diretórios precisam estar entre r"" (lida com conflitos de formatação)
diretorio_base = os.path.join(os.getcwd(), "Problema 2") # nesse caso, os dados estão na pasta 'Problema 2' diretório atual (os.getcwd())
nome_script = 'script problema 2 (6 param).mac' # arquivo dentro de diretório_base/input