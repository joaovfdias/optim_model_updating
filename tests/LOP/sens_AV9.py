from optimization.parameter import *
from sens_main import sensitivity_analysis

# AVALIAÇÃO LOP: SENSIB. À ESPESSURA DA CAMADA DE CONCRETO / MÓDULO CISALHAMENTO MADEIRA

parameters = [
            Continuous(1e6,1e9,'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            Continuous(0.25, 0.6, 'h_concreto'),
]

base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 9"

sensitivity_analysis(parameters, base_dir)