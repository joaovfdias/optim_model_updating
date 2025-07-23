from optimization.optimizer import Optimizer

import time

class Populational(Optimizer):
    def __init__(self, fitness_function, parameters, population_size):
        self.iter_label = "Iteração"
        self.global_best = None
        super().__init__(fitness_function, parameters, population_size)

    def run(self, iterations=100, status=True, log=True):
        """

                :param generations: número de gerações a serem executadas
                :param status: por padrão mostra o andamento das soluções a cada iteração, False para não mostrar
                :param log: define o registro dos resultados em planilha. True (padrão): registra os melhores indivíduos de cada iteração, "full": registra todos os indivíduos de todas as iterações. False: não cria registro.
                :return: retorna o melhor indivíduo encontrada, da qual é possível obter o fitness (.fitness), parâmetros (.param) e dados modais (.data)
        """
        self.inicio = time.time()
        self.populations = [self.initial_population()]

        full = log == "full"
        if log:
            self.create_log(full=full)
            self.add_log(0, self.populations[-1], full=full)

        if status:
            print(f"\nPopulação Inicial: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

        for iteration in range(iterations):

            new_pop = self.opt_step(iteration)

            self.evaluate_population(new_pop)
            self.populations.append(new_pop)

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if status:
                print(f"{self.iter_label} {iteration + 1}: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

            if self.tolerance(self.populations[-2], self.populations[-1]): # critério de parada, determinado com a função set_tolerance
                break

        fim = time.time()
        if log:
            self.time_log(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        self.global_best = self.global_best or self.get_best_individual(self.populations[-1])

        print(f"\nMelhor solução encontrada: Fitness = {self.global_best.fitness:.4g}, Parâmetros: {self.display_parameters(self.global_best)}")

        return self.global_best # retorna o melhor indivíduo final

    def opt_step(self, iteration):
        pass

    #GA
    def opt_step(self, iteration):
        pop = self.populations[-1]
        new_pop = self.evolve_population(pop)
        return new_pop

    #PSO
    def opt_step(self, iteration):
        new_pop = []
        w = self.w * (self.w_rate ** iteration)
        for particle in self.populations[-1]:
            new_pop.append(particle.update_particle(self.parameters, self.global_best.param, w, self.c1, self.c2))
        return new pop

    def initial_population(self):
        pop = super().initial_population()
        self.define_initial_velocity(pop)

        return pop

    def evaluate_population(self, population):
        super().evaluate_population(population)
        self.best_particles.append(self.get_best_individual(population))
        self.global_best = self.get_best_individual(self.best_particles)

