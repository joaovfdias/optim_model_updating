from ..individual import Individual

import random


class Particle(Individual):
    def __init__(self, param, fitness_function, velocity=None):
        """

        :param param: lista de valores parâmetros atreladas à partícula (posição)
        :param fitness_function: função objetivo a ser otimizada
        :param velocity: caso vazia (população inicial) será gerada randomicamente
        """
        super().__init__(param, fitness_function)
        self.velocity = velocity or [random.uniform(-1, 1) for _ in range(len(param))]

        # print(self.velocity)

    # update_particle atualiza a posição/velocidade criando uma nova partícula
    def update_particle(self, parameters, global_best_position, w, c1, c2):
        # update de posição
        pos = self.param
        new_pos = []
        for i in range(len(pos)):
            new_pos.append(pos[i] + self.velocity[i])
            new_pos[i] = max(parameters[i].lower_bound, min(new_pos[i], parameters[i].upper_bound))

        # update de velocidade
        new_vel = []
        for i in range(len(new_pos)):
            cognitive = c1 * random.random() * (global_best_position[i] - new_pos[i])
            social = c2 * random.random() * (global_best_position[i] - new_pos[i])
            new_vel.append(w * self.velocity[i] + cognitive + social)

        return Particle(new_pos, self.fitness_function, new_vel)

