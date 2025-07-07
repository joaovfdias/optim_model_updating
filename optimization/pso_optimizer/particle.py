import random

from ..individual import Individual

class Particle(Individual):
    # preciso criar uma forma de registrar o histórico de posições para usar com c1
    # inicalizar a velocidade com base no tamanho do espaço de busca
    # definir proporção no update de velocidade
    def __init__(self, param, fitness_function, velocity=None, best=None):
        """

        :param param: lista de valores parâmetros atreladas à partícula (posição)
        :param fitness_function: função objetivo a ser otimizada
        :param velocity: caso vazia (população inicial) será gerada randomicamente
        """
        super().__init__(param, fitness_function)
        self.velocity = velocity
        self.best = best

    @staticmethod
    def initial_velocity(parameters):
        vel = [random.uniform(-1,1) * (parameters[i].upper_bound - parameters[i].lower_bound) for i in range(len(parameters))]
        return vel

    def evaluate(self):
        # avalia a função caso ainda não tenha sido
        if not self.fitness:
            self.fitness, self.data = self.fitness_function(self.param)
            if not isinstance(self.data, list):  # revisar isso aqui
                self.data = [self.data]
            # preenche e registra a melhor posição com base no fitness
            self.best = self.best or [self.param, self.fitness]
            self.best = [self.param, self.fitness] if self.best[1] > self.fitness else self.best

    # update_particle atualiza a posição/velocidade criando uma nova partícula
    def update_particle(self, parameters, global_best_position, w, c1, c2):
        # update de posição
        pos = self.param
        new_pos = []
        for i in range(len(pos)):
            new_pos.append(pos[i] + self.velocity[i])
            new_pos[i] = max(parameters[i].lower_bound, min(new_pos[i], parameters[i].upper_bound))

        # atribuição de velocidade no caso inicial
        self.velocity = self.velocity or self.initial_velocity(parameters)

        # update de velocidade
        personal_best_position = self.best[0]
        new_vel = []
        for i in range(len(new_pos)):
            cognitive = c1 * random.random() * (personal_best_position[i] - new_pos[i])
            social = c2 * random.random() * (global_best_position[i] - new_pos[i])
            new_vel.append(w * self.velocity[i] + cognitive + social)

        return Particle(new_pos, self.fitness_function, new_vel, self.best)

