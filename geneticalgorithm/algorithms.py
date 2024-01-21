import random
from func.secondary_functions import BookEvolution
from base import clone 



def crossANDmut(population, crossover, mutation, cxpb, mutb):
    """
    производит мутацию и скрещивание поочередно
    """
    for i in range(0, len(population)-1):
        if random.random() < cxpb:
            population[i], population[i+1] = crossover(population[i], population[i+1])

    for i in range(0, len(population)):
        if random.random() < mutb:
            population[i] = mutation(population[i])

    return population

def crossORmut(population, crossover, mutation, cxpb, mutb):
    offspring = []

    if cxpb + mutb > 1:
        raise ("Сумма шансов скрещивание и мутации должна быть <= 1")

    for ind in population:
        r = random.random()
        if r < cxpb:
            offspring.append(crossover(ind, random.choice(population)))
        elif r < cxpb + mutb:
            offspring.append(mutation(ind))
        else:
            offspring.append(ind)

    return offspring

def classicGA(population, fitfunc, select, crossover, mutation, stats, cxpb, mutb, generations):
    bookeval = BookEvolution()

    """Классический ГА
    1.Подсчет и "установка" пригодности

    2.Запускается цикл поколение generations

        1.Проводится отбор, который полность заменяет родительскую популяцию 1к1 при этом алгоритм требуют, 
        что выбор был стохастический и возможность выбора одного и того же индивида несколько раз

        2.Клонирование т. к. из-за выбора одного и того же индивида, несколько индивидов могут ссылаться на один и тот же object

        3.Производится скрещивание с вероятностью cxpb и с вероятностью мутации mutb для формирование новой популяции

        4.Подсчет и изменение пригодности

    3.Возвращение окончательной популяции
    """
    fitneses = list(map(fitfunc, population))

    for i in range(len(population)):
        population[i].fitness.setValue(fitneses[i])

    bookeval.write(gen=0, **stats.statistics(population))

    for g in range(1, generations+1):
        offspring = select(population, len(population))
        offspring = list(map(clone, offspring))

        offspring = crossANDmut(offspring, crossover, mutation, cxpb, mutb)

        population[:] = offspring

        fitneses = list(map(fitfunc, population))

        for i in range(len(population)):
            population[i].fitness.setValue(fitneses[i])

        bookeval.write(gen=g, **stats.statistics(population))

    return population, bookeval


def classicGAElitism(population, fitfunc, select, crossover, mutation, stats, cxpb, mutb, generations, hob):
    bookeval = BookEvolution()

    fitneses = list(map(fitfunc, population))

    for i in range(len(population)):
        population[i].fitness.setValue(fitneses[i])

    hob.register(population)

    bookeval.write(gen=0, **stats.statistics(population))

    for g in range(1, generations+1):
        offspring = select(population, len(population)-hob.size)
        offspring = list(map(clone, offspring))

        offspring = crossANDmut(offspring, crossover, mutation, cxpb, mutb)

        offspring.extend(hob)
        hob.register(offspring)

        population[:] = offspring

        fitneses = list(map(fitfunc, population))

        for i in range(len(population)):
            population[i].fitness.setValue(fitneses[i])

        bookeval.write(gen=g, **stats.statistics(population))

    return population, bookeval
