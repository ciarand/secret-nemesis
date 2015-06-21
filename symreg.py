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
    return get_dataset("train.txt")

def get_testing_dataset():
    """ returns a dictionary of input => output values extracted from the
    testing data """
    return get_dataset("test.txt")

def get_dataset(filename):
    data = list()
    for line in open(filename, "r"):
        inp, out = line.rstrip().strip("()").split(" ")
        data.append((float(inp), float(out)))

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

def div(left, right):
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
    fset.addPrimitive(div, 2)

    # unary operators (arity of 1)
    fset.addPrimitive(math.sin, 1)

    # all constants are in the range of [0, 1]
    fset.addEphemeralConstant("rand0to1", lambda: random.randint(0, 1))

    fset.renameArguments(ARG0="x")

    return fset

def configure_toolbox(pset, tournsize):
    """ Creates and configures a DEAP toolbox object """

    # minimization problem, so weights are -1
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=10)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("compile", gp.compile, pset=pset)

    # population function
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation function, pass toolbox and points args
    toolbox.register("evaluate", eval_symb_reg, toolbox=toolbox, points=get_training_dataset())

    # tournament size
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    # mating strategy
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # limit mating and mutation to a tree w/ max height of 50
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))

    return toolbox

def main(num_generations=10000,
         initial_pop_num=1000,
         crossover_prob=0.9,
         mutation_prob=0.10,
         tournament_size=3):
    """ Entry point """
    random.seed()

    fset = build_fset()

    toolbox = configure_toolbox(fset, tournament_size)

    pop = toolbox.population(initial_pop_num)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   crossover_prob, mutation_prob,
                                   num_generations,
                                   stats=mstats,
                                   halloffame=hof,
                                   verbose=True)

    print(log.stream)

    winner_raw = hof[0]
    winner = toolbox.compile(winner_raw)

    margin_of_error = 0.0
    test_data = get_testing_dataset()
    for row in test_data:
        inp, out = row
        margin_of_error += abs(winner(inp) - out)

    print("The winning function was: %s" % winner_raw)
    print("With a margin of error of %f" % margin_of_error)

if __name__ == "__main__":
    main()
