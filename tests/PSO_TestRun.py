from optimization.parameter import Continuous
from optimization.pso_optimizer.pso_optimizer import PSO


# declaração da função objetivo:
def fitness_function(param):
    x, y, z = param
    return abs(x**3 + y**3 + z**3 - 42)

# parâmetros do modelo:
parameters =    [
                Continuous(-8,8, 'x'),
                Continuous(-8, 12, 'y'),
                Continuous(-25, 25, 'z')
                ]

# parâmetros do algoritmo:
w = 0.6 # inércia
w_rate = 0.99 # taxa de decaimento de inércia
c1 = 2.05 # governa a exploração da população
c2 = 2.05 # governa a convergência
init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial

population_size = 100
iteracoes = 100

# declarção do otimizador:
rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio)

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado, "full": todos os indivíduos)
log_title = None # alterar nome do arquivo gerado, se quiser
log_dir = None # alterar diretório do registro, por padrão \log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(iteracoes, log=log)