from optimization.parameter import Continuous

import os


# 1. DEFINIÇÃO DOS DADOS DO PROBLEMA

# 1.1. LISTA DE PARÂMETROS
    # lista de objetos derivados da classe Parameter
        # para esse caso, valores contínuos com limites de busca e identificador

parameters = [
    Continuous(60e9, 80e9, 'modulo_alum'),

    Continuous(0.1, 0.40, 'poisson'),
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