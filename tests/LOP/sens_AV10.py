from optimization.parameter import *
from sens_main import sensitivity_analysis

# AVALIAÇÃO LOP: PARAMETROS SENSÍVEIS + ESPESSURA + MÓDULO TRANSVERSAL MADEIRA

parameters = [# 10
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

base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 10"

sensitivity_analysis(parameters, base_dir)