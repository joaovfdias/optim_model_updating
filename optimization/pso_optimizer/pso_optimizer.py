from ..optimizer import Optimizer
from .particle import Particle

from pyDOE import lhs
import numpy as np
import time

class PSOOptimizer(Optimizer):
    def __init__(self, fitness_function, parameters, population_size, w=0.5, c1=2.0, c2=2.0):
        """

        :param fitness_function: função objetivo a ser otimizada (recebe: lista de valores dos parâmetros, retorna: fitness, [dados])
        :param parameters: lista de objetos da classe Parameters declarados com limites inferior e superior e nome
        :param population_size: número de partículas do enxame (tamanho da população)
        :param w: fator de inércia, que controla a influência da velocidade anterior no update da velocidade atual
        :param c1: coeficiente de aceleração cognitiva (ou pessoal), atrai a partícula para SUA melhor posição (solução) conhecida
        :param c2: coeficiente de aceleração social, atrai a partícula para a melhor solução GLOBAL conhecida

        O metodo de amostragem padrão está definido como 'LHS', para mudar use 'get_sampling_method'
        """
        super().__init__(fitness_function, parameters, population_size)

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.best_particles = [min(self.populations[-1], key=lambda p: p.fitness)]
        self.global_best_position = self.best_particles[-1].param[:]
        self.global_best_fitness = self.best_particles[-1].fitness

    def random_initial_population(self):
        return [Particle([p.random_value() for p in self.parameters], self.fitness_function) for _ in
                range(self.population_size)]

    def LHS_initial_population(self):
        n_dim = len(self.parameters)
        n_samples = self.population_size

        samples = lhs(n_dim, samples=n_samples)

        lower_bounds = np.array([p.lower_bound for p in self.parameters])
        upper_bounds = np.array([p.upper_bound for p in self.parameters])

        scaled_samples = lower_bounds + samples * (upper_bounds - lower_bounds)

        return [
            Particle([float(value) for value in scaled_samples[i]], self.fitness_function)
            for i in range(self.population_size)
        ]

    def run(self, itera=100, status=True, log=True):
        """

        :param itera: número de iterações a serem executadas
        :param status: por padrão mostra o andamento das soluções a cada iteração, False para não mostrar
        :param log: define o registro dos resultados em planilha. True (padrão): registra os melhores indivíduos de cada iteração, "full": registra todos os indivíduos de todas as iterações. False: não cria registro.
        :return: retorna a melhor partícula encontrada, da qual é possível obter o fitness (.fitness), parâmetros (.param) e dados modais (.data)
        """

        full = log == "full"
        if log:
            self.create_log(full=full)

        for iteration in range(itera):
            new_pop = []
            for particle in self.populations[-1]:
                new_pop.append(particle.update_particle(self.parameters, self.global_best_position, self.w, self.c1, self.c2))

                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.param

            self.populations.append(new_pop)

            self.best_particles.append(min(self.populations[-1], key=lambda p: p.fitness))

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if status:
                print(f"Iteração: {iteration + 1}, Melhor Fitness: {self.best_particles[-1].fitness}, Parâmetros: {self.best_particles[-1].param}")

        best_particle = min(self.best_particles, key=lambda p: p.fitness)

        fim = time.time()
        if log:
            self.time_log(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        print(f"\nMelhor solução encontrada: Fitness = {best_particle.fitness}, Parâmetros = {best_particle.param}")

        return best_particle