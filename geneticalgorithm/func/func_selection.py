from operator import attrgetter
import random

def randomSel(population, count):
    #Выбирает случайных count индивидов
    return [random.choice(population) for i in range(count)]

def bestSel(population, count, attr_fit='fitness'):
    #Выбирает count лучших индивидов из population
    return sorted(population, key=attrgetter(attr_fit), reverse=True)[:count]

def worstSel(population, count, attr_fit='fitness'):
    #Выбирает count хучших индивидов из population
    return sorted(population, key=attrgetter(attr_fit))[:count]

def rouletteSel(population, count, attr_fit='fitness'):
    #отбор рулеткой
    #для fitness <0, 0 меньше не подойдет
    sorted_population = sorted(population, key=attrgetter(attr_fit), reverse=True)
    sum_fit = sum(getattr(ind, attr_fit).getValue() for ind in population)
    winners = []
    for i in range(count):
        n = random.random() * sum_fit
        sum_ = 0
        for ind in sorted_population:
            sum_ += getattr(ind, attr_fit).getValue()
            if sum_ > n:
                winners.append(ind)
                break

    return winners

def tournamentSel(population, count, tournsize,  attr_fit='fitness'):
    """
    Каждый раз из популяции случайным образом отбирается несколько претендентов (от двух и более). 
    Затем, среди отобранных участников выбирается наиболее приспособленный (с наибольшим значением функции приспособленности). 
    Он и переходит в новую выборку.
    Процесс повторяется до тех пор, пока число «родителей» не станет равно размеру популяции.
    """
    winners = []
    for i in range(count):
        candidates = randomSel(population, tournsize)
        winners.append(max(candidates, key=attrgetter(attr_fit)))
    return winners

def susSel(population, count, attr_fit='fitness'):
    #Стохастическая универсальная выборка
    sorted_population = sorted(population, key=attrgetter(attr_fit), reverse=True)
    sum_fit = sum(getattr(ind, attr_fit).getValue() for ind in population)
    
    interval = sum_fit/count
    start = random.uniform(0, interval)
    roulette_arrows = [start+i*interval for i in range(count)]

    winners = []
    for ra in roulette_arrows:
        n = 0
        sum_ = getattr(sorted_population[n], attr_fit).getValue()
        while sum_ < ra:
            n += 1
            sum_ += getattr(sorted_population[n], attr_fit).getValue()
        winners.append(sorted_population[n])

    return winners

def rankedSel(population, count, attr_fit='fitness'):
    sorted_population = sorted(population, key=attrgetter(attr_fit), reverse=True)
    sum_fits = int((len(sorted_population)+1)*(len(sorted_population)/2))

    winners = []
    for i in range(count):
        n = random.random() * sum_fits
        sum_ = 0
        for k, ind in enumerate(sorted_population):
            sum_ += len(sorted_population)-k
            if sum_ > n:
                winners.append(ind)
                break

    return winners

# def rankedSel(population, count, attr_fit='fitness'):
#     sorted_population = sorted(population, key=attrgetter(attr_fit))
#     ranked = [i for i in range(len(sorted_population)) for n in range(i+1)]
    
#     winners = []
#     for i in range(count):
#         winners.append(sorted_population[random.choice(ranked)])

#     return winners

def fitnessScalingSel(population, count, interval, attr_fit='fitness'):
    """
    Развитие идеи ранжированного отбора привело к способу масштабированного отбора. 
    Здесь вместо рангов исходные значения приспособленности масштабируются к заданному интервалу значений.
    """
    sorted_population = sorted(population, key=attrgetter(attr_fit), reverse=True)
    f_min = getattr(sorted_population[-1], attr_fit).getValue()
    f_max = getattr(sorted_population[0], attr_fit).getValue()
    d_min, d_max = interval
    a = (d_min - d_max)/(f_min-f_max)
    b = d_min - a*f_min
    sum_fits = sum([getattr(ind, attr_fit).getValue()*a + b for ind in sorted_population])
    
    winners = []
    for i in range(count):
        n = random.random() * sum_fits
        sum_ = 0
        for ind in sorted_population:
            sum_ += getattr(ind, attr_fit).getValue()*a + b
            if sum_ > n:
                winners.append(ind)
                break

    return winners