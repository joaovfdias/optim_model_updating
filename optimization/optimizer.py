class Optimizer():
    def __init__(self, fitness_function, parameters):
        self.fitness_function = fitness_function
        self.parameters = parameters

    def optimize(self):
        raise NotImplementedError('Use subclasses')