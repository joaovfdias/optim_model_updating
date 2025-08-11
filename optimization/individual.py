import time


class Individual:
    def __init__(self, param, fitness_function):
        self.param = param
        self.fitness_function = fitness_function
        self.fitness = None
        self.data = None
        self.etime = None

    def __str__(self):
        param_str = ", ".join(f"{p:.4f}" for p in self.param)
        return f"Individual(param=[{param_str}], fitness={self.fitness:.4f})"

    def evaluate(self):
        if not self.fitness:
            result = self.fitness_function(self.param)
            self.fitness, self.data = result if isinstance(result, tuple) else (result, None)  # armazenar self.data apenas se o retorno da função exigir
#            self.etime = time.time()

    @staticmethod
    def compare_individuals(ind1, ind2):
        for i, _ in enumerate(ind1.param):
            if ind1.param[i] != ind2.param[i]:
                return False
        return True