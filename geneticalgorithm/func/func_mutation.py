import random


def mutFlipBit(individ, chance):

    for i in range(len(individ)):
        if random.random() < chance:
            individ[i] = int(not individ[i])

    return individ


def mutExchangeIndexes(individ, chance):
    size = len(individ)
    for i in range(size):
        if random.random() < chance:
            point = random.randint(0, size-1)
            individ[i], individ[point] = individ[point], individ[i]

    return individ


