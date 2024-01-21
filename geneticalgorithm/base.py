from copy import deepcopy


def clone(ind):
    return deepcopy(ind)#


def generate_iter(container, generator):
    """
    :container: class
    :generator: func
    """
    return container(generator())

def generate_repeat(container, n: int, object):
    """
    :container: class
    :n: len container
    :object: func
    """
    return container(object() for i in range(n))


class Fitness(object):

    weight = 1 #вес определяет будет ли ГА искать максимум или минимум

    wvalue = 0 #значение умноженное на вес

    def __init__(self, value=None) -> None:
        self.value = value

    def setValue(self, value) -> None:
        self.wvalue = value*self.weight #(значение) -> значение*вес
        self.value = value

    def getValue(self):
        return self.wvalue

    def delValues(self) -> None:
        self.wvalue = 0

    #Сравнение Fitness #########################
    def __gt__(self, other) -> bool:
        return not self.__le__(other)# >=

    def __ge__(self, other) -> bool:
        return not self.__lt__(other)# >

    def __le__(self, other) -> bool:
        return self.wvalue <= other.wvalue# <=

    def __lt__(self, other) -> bool:
        return self.wvalue < other.wvalue# <

    def __eq__(self, other) -> bool:
        return self.wvalue == other.wvalue# ==

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)# !=
    ############################################

    def __str__(self):
        return str(self.wvalue)


class Individ(list):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.fitness = Fitness()

