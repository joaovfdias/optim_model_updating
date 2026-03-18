from __future__ import annotations

import time

from optimization import Optimizer


# subclasse para funções comuns a algoritmos populacionais
class PopulationBased(Optimizer):
    def __init__(self, fitness_function, parameters, population_size):
        self.global_best = None
        super().__init__(fitness_function, parameters, population_size)

    def run(self, iterations=100, status=True, log=True):
        """
        :param itera: número de iterações a serem executadas
        :param status: por padrão mostra o andamento das soluções a cada iteração, False para não mostrar
        :param log: define o registro dos resultados em planilha. True (padrão): registra os melhores indivíduos de cada iteração, "full": registra todos os indivíduos de todas as iterações. False: não cria registro.
        :return: retorna a melhor partícula encontrada, da qual é possível obter o fitness (.fitness), parâmetros (.param) e dados modais (.data)
        """
        self.inicio = time.time()
        self.status = status

        if not self.log_history: # caso não tenha sido indicada reconstrução a partir de log anterior, gera e registra a população inicial normalmente
            self.populations.append(self.initial_population())
            full = log == "full"
            if log:
                self.add_log(0, self.populations[-1], full=full)
            if self.status:
                print(f"\nPopulação Inicial: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

        else:
            self.resume_initial_population() # reconstroi as populações iniciais com base em log anterior
            full = True # demanda registro completo

        for iteration in range(self.logged_iteration, iterations):

            new_pop = self.opt_step(iteration)

            self.evaluate_population(new_pop)
            self.populations.append(new_pop)

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if self.status:
                print(f"{self.iter_label} {iteration + 1}: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

            if self.tolerance(self.populations[-2], self.populations[-1]): # critério de parada, determinado com a função set_tolerance
                print("Execução interrompida.")
                break

        fim = time.time()
        if log:
            self.log_specs()
            self.log_time(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        self.global_best = self.global_best or self.get_best_individual(self.populations[-1])

        print(f"\nMelhor solução encontrada: Fitness = {self.global_best.fitness:.4g}, Parâmetros: {self.display_parameters(self.global_best)}")

        return self.global_best # retorna o melhor indivíduo final

    # @abstractmethod
    def opt_step(self, iteration): # definida nos algoritmos específicos
        pass
