from optimization.parameter import Continuous
from optimization.ga_optimizer import GA as GA


def fitness_function(param):
    x, y, z = param
    return abs(x ** 3 + y ** 3 + z ** 3 - 42)

# arquivo log
log = "full"

# parâmetros do algoritmo
parameters = [Continuous(-8,8, 'x'), Continuous(-8, 12, 'y'), Continuous(-25, 25, 'z')]

elitism_rate = 0.10
crossover_rate = 0.60
mutation_strength = 0.10 # no tipo de mutação "gaussian" para variáveis contínuas, alterar essa taxa diretamente (define a faixa percentual em que os paramêtros podem variar)

population_size = 10
generations = 10

# chamada
rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength)
#rodada.set_log(log_name, log_dir)
best = rodada.run(generations, log=log)
