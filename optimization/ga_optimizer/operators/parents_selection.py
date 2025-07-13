import random

def select_parent(self, population, type="roulette_wheel", truncation_rate=0.3):
    """selects two parent individuals to pass through the crossover process"""
    selection_method = {
        "roulette_wheel": self.roulette_wheel,
        "tournament": self.tournament_selection,
        "random": self.random_selection,
        "truncation": self.truncation_selection,
        "ranking": self.ranking_selection,
    }  # selection methods avaiable
    return selection_method[type](population, truncation_rate)


def roulette_wheel(self, population, truncation_rate):
    """Selects randonly one individual from the population. The probability of an individual to be chosen is as high as its fit"""
    fitnesses = [population[i].fitness for i in range(len(population))]
    total = sum(fitnesses)
    inversion = [total - fit for fit in fitnesses]  # for the probability to be higher for smaller fitness_values
    total_inversion = sum(inversion)
    cumulative_probabilities = []
    cumulative_weight = 0
    for fitness in fitnesses:
        cumulative_weight += (total - fitness) / total_inversion
        cumulative_probabilities.append(cumulative_weight)
    rand = random.random()
    for i, individual in enumerate(population):
        if rand <= cumulative_probabilities[i]:
            return individual
    raise IndexError('Individual not found in roulette wheel method.')


def tournament_selection(self, pop, truncation_rate):
    """Selects randomly 3 individuals from the population and returns the one with best fit"""
    competitors = [pop[random.randint(0, len(pop) - 1)] for _ in range(3)]  # selects 3 randomly
    return min(competitors, key=lambda x: x.fitness)


def random_selection(self, pop, truncation_rate):
    """Parents are selected randomly"""
    return random.choice(pop)


def truncation_selection(self, pop, truncation_rate):
    """selects randomly one individual in the elite, considering the given truncation rate"""
    elite = sorted(pop, key=lambda x: x.fitness)[:(int(len(pop) * truncation_rate))]
    return elite[random.randint(0, len(elite) - 1)]


def ranking_selection(self, pop, truncation_rate):
    """selects randomly one individual considering higher probability for better fit, but with controled probability variation"""
    sort_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    probabilities = 0
    cumulated_probabilities = []
    rand = random.random()
    ranking = [individual for individual in sort_pop]
    for i in range(len(ranking)):
        probabilities += (i + 1) / len(ranking)
        cumulated_probabilities.append(probabilities)
    for i, individual in enumerate(ranking):
        if rand <= cumulated_probabilities[i]:
            return individual
    raise IndexError('Individual not found in ranking selection method.')
