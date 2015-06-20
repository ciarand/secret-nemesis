"""
symreg.py provides a simple symbolic regression program that solves assignment
1 for CSUF's Summer 2015 CPSC 481 course.
"""

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def get_training_dataset():
    """ returns a dictionary of input => output values extracted from the
    training data """
    data = list()
    for line in open("train.dat", "r"):
        data.append(line.rstrip().split(" "))

    return data

def eval_symb_reg(individual, points, toolbox):
    """ Takes an individual, randomly generated function and compares it with
    each row in the training dataset. Returns a decimal value indicating the
    margin of errors (higher is worse).
    """

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    errors = 0.0
    for row in points:
        inp, out = row
        errors += abs(func(inp) - out)

    return (errors, )

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def build_fset():
    """ Fset = {+, -, *, /, sin(x)} """

    # our function only expects 1 input, x
    fset = gp.PrimitiveSet("main", 1)

    # binary operators (arity of 2)
    fset.addPrimitive(operator.add, 2)
    fset.addPrimitive(operator.sub, 2)
    fset.addPrimitive(operator.mul, 2)
    fset.addPrimitive(protectedDiv, 2)

    # unary operators (arity of 1)
    fset.addPrimitive(math.sin, 1)

    # all constants are in the range of [0, 1]
    fset.addEphemeralConstant("rand0to1", lambda: random.randint(0, 1))

    return fset

def configure_toolbox(pset):
    """ Creates and configures a DEAP toolbox object """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=100)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_symb_reg, toolbox=toolbox, points=get_training_dataset())

    # boilerplatey looking functions (TODO: investigate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox

def main():
    """ Entry point """
    random.seed(102)

    fset = build_fset()

    toolbox = configure_toolbox(fset)

    pop = toolbox.population(5)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.9, 0.05, 1, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()

    print(log.stream)
