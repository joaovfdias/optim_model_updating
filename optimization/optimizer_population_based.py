from __future__ import annotations

import time

from parameter import Parameter
from optimization import Optimizer, Individual
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any


# subclasse para funções comuns a algoritmos populacionais
class PopulationBased(Optimizer):
    """
    Subclasse de Optimizer para comportar o fluxo de execução comum a algoritmos populacionais.
    """
    def __init__(self, fitness_function: Callable[[list[float]], tuple[float, dict[str, Any]]], parameters: list[Parameter], population_size: int):
        """

        :param fitness_function: Função que recebe os parâmetros de modelo, avalia as métricas e retorna fitness (float) + dados adicionais (dict).
        :param parameters: Variáveis que se deseja calibrar: lista de objetos da classe Parameter com os devidos atributos declarados.
        :param population_size: Tamanho da população. Se não especificado, default para 10 vezes o número de variáveis.
        """
        self.global_best = None
        super().__init__(fitness_function, parameters, population_size)

    def run(self, iterations: int | None = None, status: bool = True, log: bool | str = True) -> Individual:
        """
        Executa a otimização pela quantidade de iterações/gerações definida. Para definir critérios de parada, usar set_tolerance anteriormente a chamada desta função.

        :param iterations: Número de iterações/gerações máximo a serem executadas. Se não definido, usa 10 vezes o número de variáveis.
        :param status: Booleano para mostrar ou não o andamento de cada iteração no terminal.
        :param log: Define o registro dos resultados em planilha. * True (padrão): registra os melhores indivíduos de cada iteração. * "full": registra todos os indivíduos de todas as iterações. * False: não cria registro.

        :return: O melhor indivíduo/partícula encontrado, do qual é possível obter o fitness (.fitness), parâmetros (.param) e dados adicionais (.data)
        """
        self.inicio = time.time()
        self.status = status
        iterations = iterations or 10 * len(self.parameters)

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
    def opt_step(self, iteration: int) -> list[Individual]: # definida nos algoritmos específicos
        """
        Etapa de otimização específica a cada algoritmo.

        :param iteration: Número da iteração/geração atual.

        :return: Nova população gerada para a iteração (lista de instâncias de Individual).
        """
        pass
