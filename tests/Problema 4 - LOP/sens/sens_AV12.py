from optimization.parameter import *
from sens_main import sensitivity_analysis

# AVALIAÇÃO LOP: TODOS

parameters = [ # analise 12
    Continuous(28e9, 34e9, 'modulo_concreto'),
    Continuous(0.04, 0.06, 'h_concreto'),

    Continuous(12e9, 18e9, 'modulo_madeira'),

    Continuous(200e9, 220e9, 'modulo_cordoalhas'),

    Continuous(5e7, 5e8, 'kv'),
    Continuous(5e7, 5e8, 'kh'),

    Continuous(1e7, 1e9, 'GXY'),
    Continuous(1e7, 1e9, 'GYZ'),
    Continuous(1e7, 1e9, 'GXZ')
]

base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 12"

sensitivity_analysis(parameters, base_dir)