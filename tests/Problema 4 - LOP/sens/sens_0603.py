from optimization.parameter import *
from sens_main import sensitivity_analysis

parameters = [  # analise 12
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(0.1, 0.49, 'poisson_concreto'),
            Continuous(0.02, 0.06, 'h_concreto'),

            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(0.1, 0.49, 'poisson_madeira'),

            Continuous(150e9, 250e9, 'modulo_perfis'),
            Continuous(0.1, 0.49, 'poisson_perfis'),

            Continuous(150e9, 250e9, 'modulo_cordoalhas'),
            Continuous(0.1, 0.49, 'poisson_cordoalhas'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt'),

            Continuous(1e6, 1e9, 'ey_wood'),  # esperado 10 a 500 MPa
            Continuous(1e7, 1e9, 'GXY'),  # 600-900 MPa  6e8
            Continuous(1e6, 1e8, 'GYZ')  # 50-150 MPa  5e7
            # Continuous(1e7, 5e7, 'GXZ')
        ]

base_dir = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 4\input\antigo (15 GERAL +EY -GXZ)"
local_dir = r"C:\Users\thiag\Documentos (Local)\Problema 4 (2026)"

sensitivity_analysis(parameters, base_dir, local_dir=local_dir)