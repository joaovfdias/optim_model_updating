import random


class Parameter:
    def __init__ (self, key):
        self.key = key

# é preciso estudar e testar variáveis binárias e de estado
class Continuous(Parameter):
    def __init__ (self, lower_bound, upper_bound, key):
        super().__init__(key)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.search_space = upper_bound - lower_bound

    def random_value(self):
        return random.uniform(self.lower_bound,self.upper_bound)

    def check_bounds(self, value):
        """
        Verifica se um valor está dentro dos limites do parâmetro.
        :param value: Valor a ser verificado.
        :return: True se o valor está dentro dos limites, False caso contrário.
        """
        return self.lower_bound <= value <= self.upper_bound

class Integer(Parameter):
    def __init__ (self, lower_bound: int, upper_bound: int, key: list[str]):
        super().__init__(key)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.search_space = upper_bound - lower_bound

    def random_value(self):
        return random.randint(self.lower_bound,self.upper_bound)

    def check_bounds(self, value):
        """
        Verifica se um valor está dentro dos limites do parâmetro.
        :param value: Valor a ser verificado.
        :return: True se o valor está dentro dos limites, False caso contrário.
        """
        return self.lower_bound <= value <= self.upper_bound

class Binary(Parameter):
    def __init__ (self, key):
        super().__init__(key)

    def random_value(self):
        return random.getrandbits(1)

class State(Parameter):
    """Estados indicados por um valor inteiro"""
    def __init__ (self, lower_bound, upper_bound, key):
        super().__init__(key)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.search_space = abs(upper_bound-lower_bound)+1
        self.search_list = list(range(lower_bound, upper_bound+1))

    def random_value(self):
        return random.randint(self.lower_bound,self.upper_bound)