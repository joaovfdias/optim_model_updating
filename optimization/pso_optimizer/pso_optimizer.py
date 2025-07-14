import time

from ..optimizer import Optimizer


class PSO(Optimizer):
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

        self.best_particles = None
        self.global_best = None


    def define_initial_velocity(self, population):
        for particle in population:
            particle.initial_velocity(self.parameters, self.init_vel_ratio)

    def run(self, itera=100, status=True, log=True):
        """

        :param itera: número de iterações a serem executadas
        :param status: por padrão mostra o andamento das soluções a cada iteração, False para não mostrar
        :param log: define o registro dos resultados em planilha. True (padrão): registra os melhores indivíduos de cada iteração, "full": registra todos os indivíduos de todas as iterações. False: não cria registro.
        :return: retorna a melhor partícula encontrada, da qual é possível obter o fitness (.fitness), parâmetros (.param) e dados modais (.data)
        """

        self.populations = [self.initial_population()]
        self.define_initial_velocity(self.populations[-1])
        self.best_particles = [self.get_best_individual(self.populations[-1])]
        self.global_best = self.best_particles[-1]


        full = log == "full"
        if log:
            self.create_log(full=full)
            self.add_log(0, self.populations[-1], full=full)

        if status:
            print(f"População Inicial: Melhor Fitness = {self.best_particles[-1].fitness:.4g}, Parâmetros: {self.display_parameters(self.best_particles[-1])}")

        for iteration in range(itera):
            new_pop = []
            w = self.w * (self.w_rate ** iteration)
            for particle in self.populations[-1]:
                new_pop.append(particle.update_particle(self.parameters, self.global_best.param, w, self.c1, self.c2))

            self.evaluate_population(new_pop)
            self.populations.append(new_pop)

            self.best_particles.append(min(self.populations[-1], key=lambda p: p.fitness))
            self.global_best = self.get_best_individual(self.best_particles)

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if status:
                print(f"Iteração {iteration + 1}: Melhor Fitness = {self.best_particles[-1].fitness:.4g}, Parâmetros: {self.display_parameters(self.best_particles[-1])}")

                #print(f"Iteração {iteration + 1}: Melhor Fitness = {self.best_particles[-1].fitness}, Parâmetros = {self.best_particles[-1].param}")

        fim = time.time()
        if log:
            self.time_log(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        print(f"\nMelhor solução encontrada: Fitness = {self.global_best.fitness:.4g}, Parâmetros: {self.display_parameters(self.global_best)}")

        return self.global_best