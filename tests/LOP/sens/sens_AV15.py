from optimization.parameter import *
from sens_main import sensitivity_analysis

# AVALIAÇÃO LOP: AVALIANDO Gs fixos e Ey variavel

parameters = [ # analise 15
    Continuous(0.04, 0.06, 'h_concreto'),

    Continuous(1e7, 2e9, 'ey_wood') # esperado 10 a 500 MPa
]

base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 15"

script_name = "scriptLOP.mac"

sensitivity_analysis(parameters, base_dir)