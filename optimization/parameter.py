import random


class Parameter:
    def __init__(self, lower_bound, upper_bound, key=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.key = key

    def random_value(self):
        return random.uniform(self.lower_bound, self.upper_bound)


class Continuous(Parameter):
    def __init__(self, lower_bound, upper_bound, key=None):
        super().__init__(lower_bound, upper_bound, key)

    def random_value(self):
        return random.uniform(self.lower_bound, self.upper_bound)

    def check_bounds(self, value):
        """
        Verifica se um valor está dentro dos limites do parâmetro.

        :param value: Valor a ser verificado.
        :return: True se o valor está dentro dos limites, False caso contrário.
        """
        return self.lower_bound <= value <= self.upper_bound


class Binary(Parameter):
    def __init__(self, lower_bound, upper_bound, key=None):
        super().__init__(lower_bound, upper_bound, key)

    def random_value(self):
        return random.getrandbits(1)


class State(Parameter):
    """Estados indicados por um valor inteiro"""

    def __init__(self, lower_bound, upper_bound, key=None):
        super().__init__(lower_bound, upper_bound, key)

    def random_value(self):
        return random.randint(self.lower_bound, self.upper_bound)
