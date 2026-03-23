from optimization.parameter import *
from sens_main import sensitivity_analysis

# AVALIAÇÃO LOP: AVALIANDO GXZ ALTO E FIXO (~900MPa)

parameters = [ # analise 14
    Continuous(0.04, 0.06, 'h_concreto'),

    Continuous(1e7, 1e9, 'GXY'), # 600-900 MPa  6e8
    Continuous(1e6, 1e8, 'GYZ') # 50-150 MPa  5e7
]

base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 14"

script_name = "scriptLOP.mac"

sensitivity_analysis(parameters, base_dir)