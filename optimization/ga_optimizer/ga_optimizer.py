import time

from ..optimizer import Optimizer, PopulationBased
from ..parameter import *
from ..individual import Individual
from .operators.parents_selection import select_parent
from .operators.crossover import crossover_methods, crossover
from .operators.mutation import mutate


class GA(PopulationBased):

    def __init__(self, fitness_function, parameters, population_size, elitism_rate=0.1, crossover_rate=0.6, mutation_strength=0.05):
        """

        :param fitness_function: função objetivo a ser otimizada (função que recebe lista de valores dos parâmetros e retorna: fitness, [dados]
        :param parameters: lista de objetos da classe Parameters declarados com limites inferior e superior e nome
        :param population_size: número de indvíduos por geração (tamanho da população)
        :param elitism_rate:
        :param crossover_rate:
        :param mutation_strength:
        """
        super().__init__(fitness_function, parameters, population_size)

        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = [1 for _ in range(len(parameters))] # probabilidade de mutar (Gaussiana), pode ser definida com "set"
        self.mutation_strength = mutation_strength # força de mutação, define a faixa de variação

        self.selection_method = "tournament" # pode ser definida com "set"
        self.truncation_rate = 0.3
        self.crossover_type = "uniform"
        self.mutation_type = self.generate_mutation_type()

        self.iter_label = "Geração"


    def set_selection_parents(self, selection_type, truncation_rate=0.3):
        """
                Permite ao usuário definir o tipo de seleção de pais e o valor de 'truncation_rate'.
        """
        self.selection_method = selection_type
        self.truncation_rate = truncation_rate

    def set_crossover(self, crossover_type):
        """
                Permite ao usuário definir o tipo de crossover, validando se o tipo é permitido.
        """
        if crossover_type in crossover_methods:
            self.crossover_type = crossover_type
        else:
            raise ValueError(f"Crossover type '{crossover_type}' inválido. Tipos válidos: {list(crossover_methods.keys())}")

    def set_mutation_rate(self, mutation_rate):
        """
        Permite alterar a taxa de mutação, que define a probabilidade de mutar no tipo "gaussian"
        """
        self.mutation_rate = [mutation_rate for _ in range(len(self.parameters))] # limita a mesma a taxa para todos parâmetros [GIEDRE revisar versão antiga no final]

    def generate_mutation_type(self):
        """atributes the mutation type as Gaussian to all continuous parameters, as bitflip to binary and as random to state parameters"""
        mutation_type = []
        for par in self.parameters:
            if type(par) == Continuous:
                mutation_type.append("gaussian")
            elif type(par) == Binary:
                mutation_type.append("bitflip")
            elif type(par) == State:
                mutation_type.append("random")
        return mutation_type


    def evolve_population(self, population):
        # elite
        elite_size = round(self.elitism_rate * self.population_size) # quantidade de indivíduos preservados
        #elite_size = 1 if not elite_size and self.elitism_rate else elite_size # garante 1 indíviduo na elite (apenas) para casos em que o arredondamento zera
        population.sort(key=lambda x: x.fitness) # rankeando a população

        new_population = population[:elite_size]

        while len(new_population) < self.population_size:
            # seleciona pais diferentes entre si
            parent1 = select_parent(self.populations[-1], self.selection_method, self.truncation_rate)
            parent2 = select_parent(self.populations[-1], self.selection_method, self.truncation_rate)
            while parent1 == parent2:
                parent2 = select_parent(self.populations[-1], self.selection_method, self.truncation_rate)

            # crossover
            if random.random() < self.crossover_rate: #avaliação da ocorrência ou não
                child = crossover(parent1, parent2, self.crossover_type)
                if Individual.compare_individuals(child, parent1) or Individual.compare_individuals(child, parent2):
                    child = mutate(child, self.parameters, self.mutation_type, self.mutation_rate, self.mutation_strength)

            # mutação
            else:
                child = mutate(parent1, self.parameters, self.mutation_type, self.mutation_rate, self.mutation_strength) if random.random() > 0.5 else mutate(parent2, self.parameters, self.mutation_type, self.mutation_rate, self.mutation_strength)

            new_population.append(child)

        # retonar a população atual garantindo o tamanho
        return new_population[:self.population_size]

    # etapa de execução
    def opt_step(self, iteration): # função run geral foi movida para subclasse PopulationBased herdada
        pop = self.populations[-1]
        new_pop = self.evolve_population(pop)
        return new_pop



    # as funções de 'setup' a seguir precisam ser revisadas (Giedre)

    # def setup_mut_type(self, mut_type, option, par_type=None, change = None): #verify
    #     """(type, chosen option, type to be changed(if necessary), position(if necessary). Modifies the mutation types. Options are:
    #     "list": Changes the mutation type of all parameters by sending a list of mutation type corresponding to the relative parameter
    #     "same": Changes the mutation type of all parameters of the given type to a given value (ex.: all the continuous to gaussian)
    #     "change_one": Changes the mutation type on the given position to the given type (e.: position 0 to bitflip)"""
    #     match option:
    #         case "list": #a list with different values
    #             self.mutation_type = mut_type
    #         case "same": #one mutation rate for all parameters
    #             match par_type:
    #                 case "continuous":
    #                     for i, par in enumerate(self.parameters):
    #                         self.mutation_type[i] = mut_type if type(par) == continuous else self.mutation_type[i]
    #                 case "binary":
    #                     for i, par in enumerate(self.parameters):
    #                         self.mutation_type[i] = mut_type if type(par) == binary else self.mutation_type[i]
    #                 case "state":
    #                     for i, par in enumerate(self.parameters):
    #                         self.mutation_type[i] = mut_type if type(par) == state else self.mutation_type[i]
    #         case "change_one": #one mutation rate for all parameters
    #             self.mutation_type[change] = mut_type
    #
    # def setup_mut_rate(self, mut_rate, option, change = None):
    #         """(rate, chosen option, position(if necessary). Modifies the mutation rates. Options are:
    #         "list": Change all values by sending a list of mutation rates corresponding to the relative parameter
    #         "same": Change the mutation rates of all parameter to a given value
    #         "change_one": Changes the mutation rate on the given position to the given value"""
    #         match option:
    #             case "list": #a list with different values
    #                 self.mutation_rate = mut_rate
    #             case "same": #one mutation rate for all parameters
    #                 self.mutation_rate.clear()
    #                 [self.mutation_rate.append(mut_rate) for p in self.parameters]
    #             case "change_one": #one mutation rate for all parameters
    #                 self.mutation_rate[change] = mut_rate