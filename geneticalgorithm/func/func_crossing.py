import random



def crossVoid(parent_1, parent_2):
    return parent_1, parent_2

def crossOnePoint(parent_1, parent_2):
    size = len(parent_1)
    point = random.randint(1, size-1)
    parent_1[point:], parent_2[point:] = parent_2[point:], parent_1[point:]

    return parent_1, parent_1

def crossTwoPoint(parent_1, parent_2):
    size = len(parent_1)
    point1 = random.randint(1, size)
    point2 = random.randint(1, size-2)

    if point1 >= point2:
        point2, point1 = point1, point2
    else:
        point2 += 1

    parent_1[point1:point2], parent_2[point1:point2] = parent_2[point1:point2], parent_1[point1:point2]

    return parent_1, parent_2

def crossUniform(parent_1, parent_2, chance=0.5):
    size = len(parent_1)
    for gen in range(size):
        if random.random() < chance:
            parent_1[gen], parent_2[gen] = parent_2[gen], parent_1[gen]

    return parent_1, parent_2

def crossBlend(parent_1, parent_2, alpha):
    #идея https://hexdocs.pm/genex/operators-crossover.html
    for i in range(len(parent_1)):
        gamma = (1 + (2 * alpha)) * random.random() - alpha
        parent_1[i] = (((1 - gamma) * parent_1[i]) + (gamma * parent_2[i]))
        parent_2[i] = ((gamma*parent_1[i]) + (1 - gamma)*parent_2[i])

    return parent_1, parent_2

def crossSimulatedBinary(parent_1, parent_2, eta):
    for i in range(len(parent_1)):
        u = random.random()
        beta = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta+1))
        parent_1[i] = 0.5*((1 + beta)*parent_1[i] + (1 - beta)*parent_2[i])
        parent_2[i] = 0.5*((1 - beta)*parent_1[i] + (1 + beta)*parent_2[i])
    
    return parent_1, parent_2


# УЛУЧШИТЬ ###############################################
def crossOrder(parent_1, parent_2):

    size = len(parent_1)
    point1, point2 = random.sample(range(size-1), 2)

    if point1 > point2:
        point1, point2 = point2, point1

    child1 = ["False" for i in range(point1)] + [parent_2[i] for i in range(point1, point2)] + ["False" for i in range(point2, size)]
    child2 = ["False" for i in range(point1)] + [parent_1[i] for i in range(point1, point2)] + ["False" for i in range(point2, size)]

    gens1 = [gen for gen in parent_1 if gen not in child1]
    gens2 = [gen for gen in parent_2 if gen not in child2]

    k = 0
    for i in range(point2, size+point2):
        if child1[i%size] == "False":
            child1[i%size] = gens1[k]
            child2[i%size] = gens2[k]
            k += 1

    return child1, child2
#######################################################