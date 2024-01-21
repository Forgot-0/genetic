import random 
from func.secondary_functions import *
from func.func_crossing import *
from func.func_selection import *
from func.func_mutation import *
from base import *
from algorithms import classicGAElitism
from functools import partial


BIT_LEN = 2
POPULATION_LEN = 100
CHANCE_CROSSOVER = 1
CHANCE_MUTATION_INDIVID = 0.0
GENERATIONS = 100 #поколения
CHANCE_MUTATION_GEN = 0.2


def himmelblau(ind):#функция для подсчета приспасобленности
    x, y = ind
    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

Fitness.weight = -1 #Указываем будет ли ГА искать максимум или минимум (-1 миниму, а 1 максимум)

creator_gen = lambda: random.uniform(-5, 5)

creator_individ = partial(generate_repeat, Individ, BIT_LEN, creator_gen)

creator_population = partial(generate_repeat, list, POPULATION_LEN, creator_individ)

population = creator_population()

for ind in population:
    getattr(ind, 'fitness').setValue(himmelblau(ind))

select =  partial(tournamentSel, tournsize=3)
crossover = partial(crossBlend, alpha=0.5)
mutation = partial(mutExchangeIndexes, chance=0.0)

stats = MoreStatistics([Statistic(), Statistic(key=len)])

population, bookeval = classicGAElitism(population, himmelblau, select, crossover, mutation, 
   stats, CHANCE_CROSSOVER, CHANCE_MUTATION_GEN, GENERATIONS, Bests(5))

print(max(population, key=attrgetter('fitness')), himmelblau(max(population, key=attrgetter('fitness'))))

print(bookeval.get("gen"), bookeval.sections['0'])


"""right answer
(3.0; 2.0), (-2.805118; 3.131312), (-3.779310; -3.283186), (3.584458; -1.848126)
"""



def fitness(x, y):
    return x*y