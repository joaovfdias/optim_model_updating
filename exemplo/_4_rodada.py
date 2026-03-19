from optimization.pso_optimizer.pso_optimizer import PSO # importação do executor do algoritmo

from _1_definir_dados import parameters, keys, diretorio_base as base_dir # importação dos dados declarados para o problema
from _2_definir_ansys import ansys # importação do objeto Ansys
from _3_definir_funcao import fitness_function # função que pontua cada indivíduo
# OS DADOS ACIMA PODEM SER DECLARADOS NO MESMO ARQUIVO

import os
from datetime import datetime


# 4. CONFIGURAÇÃO DO ALGORITMO E CHAMADA DA RODADA
# EXEMPLO PARA PSO

# caso queira especificar o local de salvamento do registro e o nome do arquivo
log_dir = None
log_title = None

# parâmetros do algoritmo:
w = 0.6 # proporção da velocidade atual que participa da próxima
w_rate = 0.99 # taxa de decaimento de inércia por iteração
c1 = 2.05 # influencia a exploração individual
c2 = 2.05 # influencia a convergência para o mínimo do grupo
init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial

population_size = len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
iterations = round(10*len(keys)) # quantidade de iterações (suficientemente grande para a convergência do algoritmo)


# declaração do otimizador:
rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio) # objeto otimizador
rodada.set_tolerance(fit_rel = 10e-3, patience = 10) # critério de parada
rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
log_dir = log_dir or os.path.join(base_dir, 'log', 'runs', 'PSO')
log_title = log_title or f"PSO_pop({population_size})_iter({iterations})_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
rodada.set_log(log_title, log_dir, False)

# chamada:
try:
    best = rodada.run(iterations, log=log)
finally:
    try:
        ansys.mapdl.exit(force=True)
    except:
        pass

print(f"\nRodada finalizada.")