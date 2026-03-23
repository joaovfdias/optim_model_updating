import time

from ..optimizer import Optimizer
from ..optimizer_population_based import PopulationBased


class PSO(PopulationBased):
    def __init__(self, fitness_function, parameters, population_size, w=0.6, w_rate=0.99, c1=2.0, c2=2.0, init_vel_ratio=0.20):
        """

        :param fitness_function: função objetivo a ser otimizada (função que recebe lista de valores dos parâmetros e retorna: fitness, [dados]
        :param parameters: lista de objetos da classe Parameters declarados com limites inferior e superior e nome
        :param population_size: número de partículas do enxame (tamanho da população)
        :param w: fator de inércia, que controla a influência da velocidade anterior no update da velocidade atual
        :param w_rate: controla o decaimento da inércia ao longo das iterações para comportamento final refinado
        :param c1: coeficiente de aceleração cognitiva (ou pessoal), atrai a partícula para SUA melhor posição (solução) conhecida
        :param c2: coeficiente de aceleração social, atrai a partícula para a melhor solução GLOBAL conhecida
        :param init_vel_ratio: define a faixa (%) do espaço de busca na qual a velocidade inicial pode ser definida, negativa ou positiva

        O metodo de amostragem padrão está definido como 'LHS', para mudar use 'get_sampling_method'
        """
        super().__init__(fitness_function, parameters, population_size)

        self.w = w
        self.w_rate = w_rate
        self.c1 = c1
        self.c2 = c2
        self.init_vel_ratio = init_vel_ratio

        self.best_particles = []
        self.global_best = None


    # metodos sobrescritos:
    def evaluate_population(self, population):
        super().evaluate_population(population)
        self.best_particles.append(self.get_best_individual(population))
        self.global_best = self.get_best_individual(self.best_particles)

    def initial_population(self):
        pop = super().initial_population()
        self.define_initial_velocity(pop)
        return pop

    # metodo exclusivo:
    def define_initial_velocity(self, population):
        for particle in population:
            particle.initial_velocity(self.parameters, self.init_vel_ratio)

    # etapa de execução
    def opt_step(self, iteration): # função run geral foi movida para subclasse PopulationBased herdada
        new_pop = []
        w = self.w * (self.w_rate ** iteration)
        for particle in self.populations[-1]:
            new_pop.append(particle.update_particle(self.parameters, self.global_best.param, w, self.c1, self.c2))
        return new_pop

    @property
    def specs(self):
        return {"sampling method": self.sampling_method,
                "initial velocity ratio": self.init_vel_ratio,
                "inertia weight (w)": self.w,
                "inertia decay rate": self.w_rate,
                "cognitive coefficient (c1)": self.c1,
                "social coefficient (c2)": self.c2
                }