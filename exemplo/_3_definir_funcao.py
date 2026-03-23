from utils.fitness_function import fitness_function_ansys

from _1_definir_dados import keys # lista de identificadores
from _2_definir_ansys import ansys # objeto Ansys


# 2. DEFINIR AS MÉTRICAS QUE PONTUAM CADA MODELO
    # nesse caso, criei uma "function factory" que retorna entre 2 presets de funções que usam dados modais
        # para esse exemplo, usamos dados de frequências naturais e modos de vibração

# preset escolhido e os dados adicionais necessários (ponderamento de cada termo)
preset = 1
wf = 1
wm = 1

# definição da função objetivo
fitness_function = fitness_function_ansys(keys, ansys, preset, wf=wf, wm=wm)
