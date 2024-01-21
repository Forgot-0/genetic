from copy import deepcopy
from functools import partial
from operator import attrgetter
import numpy


class Statistic(object):
    def __init__(self, key=lambda ind: ind.fitness.value) -> None:
        self.key = key
        self.functions = {'mean':self.mean, 'max': self.max, 'min': self.min}
    
    def mean(self, values):
        return numpy.mean(values)

    def max(self, values):
        return numpy.max(values)

    def min(self, values):
        return numpy.min(values)
    
    def add(self, name, func, *args, **kwargs):
        self.functions[name] = partial(func, *args, **kwargs)

    def statistics(self, data):
        """
        1. На вход подается популяция
        2. Формируется список со значениями, получаемыми по self.key (по умолчанию вытаскивает value)
        3. Формируется словарь где будут храниться пара имя функции – функция(values)
        """
        values = list(self.key(elem) for elem in data)

        entry = dict()
        for key, func in self.functions.items():
            entry[key] = func(values)

        return entry


class MoreStatistics(list):
    """
    a= Statistic(lambda ind: ind.fitness.value)
    b= Statistic(sum)
    mstatistic= MoreStatistics([a, b])
    stats = mstatistic.statistics(popul..)
    print stats
    Пример: 
    {'0': {'mean': 3.5, 'max': 4, 'min': 3}, '1': {'mean': 20.5, 'max': 23, 'min': 18}}
    0 - a
    1 - b
    """

    def add(self, *names):
        for i in range(len(names)):
            self.append(names[i])

    def statistics(self, data):
        
        statistic = {}
        for num, elem in enumerate(self):
            statistic[str(num)] = (elem.statistics(data))

        return statistic


class BookEvolution(list):

    def __init__(self):
        self.sections = {}#Секциям присваивается номера в соответсвие с их последовадельностью передачи в метод write

    def write(self, **infos):
        
        for key, value in list(infos.items()):
            if isinstance(value, dict):
                if key not in self.sections.keys():
                    be= BookEvolution()
                    be.write(**value)
                    self.sections[key] = be
                else:
                    self.sections[key].write(**value)
                del infos[key]

        self.append(infos)

    def get(self, *names):
        """
        на вход имена ключей в словаре
        возвращает кортеж со списками значений по ключам
        """
        return tuple([entry.get(name, None) for entry in self] for name in names)


class Bests(object):

    def __init__(self, size) -> None:
        self.size = size
        self.bests = list()

    def register(self, population):
        if (len(self.bests) == 0) and (self.size != 0):
            self.bests.insert(0, deepcopy(population[0]))

        for ind in population:
            if ind.fitness > self.bests[0].fitness or len(self.bests) < self.size:
                for best in self.bests:
                    if best == ind:
                        break
                else:
                    if len(self.bests) >= self.size:
                        self.bests.pop(0)
                    self.bests.append(ind)

        self.bests = sorted(self.bests, key=attrgetter('fitness'))
    
    def __str__(self) -> str:
        return str(self.bests)

    def __iter__(self):
        return iter(self.bests)

if __name__ == "__main__":
    # p = [Individ([6, 2, 4, 5, 6]), Individ([0, 2, 4, 6, 6])]
    # p[0].fitness.setValue(3)
    # p[1].fitness.setValue(4)
    # hob = HallofBest(5)
    # hob.register(p)
    # p.extend(hob)
    # print(p, (hob.size))
    pass