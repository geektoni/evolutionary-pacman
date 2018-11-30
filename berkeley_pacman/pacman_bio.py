# util
import numpy as np
import sys
from random import Random
from time import time

# Berkeley Pac-Man
import pacman
import textDisplay
import graphicsDisplay

# Keras
from keras.models import Sequential
from keras.layers import Dense

# inspyred
from inspyred import ec
from inspyred.ec import terminators
from inspyred.ec import selectors
from inspyred.ec import variators

# model save/load
import json
from keras.models import model_from_json


def generate_model():
    model = Sequential()
    model.add(Dense(10, input_shape = (20,), activation = "relu"))
    model.add(Dense(5, activation = "softmax"))
    return model


def generate_candidate(random, args):
    nn_model = args.get("nn_model")
    #weights = [np.random.sample(w.shape) for w in nn_model.get_weights()]
    weights = [np.random.uniform(-1, 1, w.shape) for w in nn_model.get_weights()]
    return weights


def evaluate_candidates(candidates, args):
    nn_model = args["nn_model"]
    cmd_line_args = args["cmd_line_args"]
    candidates_fitness = []
    for candidate in candidates:
        nn_model.set_weights(candidate)
        pacmanType = pacman.loadAgent("BioAgent", True)
        cmd_line_args['pacman'] = pacmanType(nn_model = nn_model)
        cmd_line_args['display'] =  textDisplay.NullGraphics()
        #cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
        games = pacman.runGames(**cmd_line_args)
        candidate_fitness = 0
        for game in games:
            candidate_fitness += game.state.getScore()
        candidates_fitness.append(candidate_fitness)
    print(candidates_fitness)
    return candidates_fitness


if __name__ == '__main__':
    cmd_line_args = pacman.readCommand(sys.argv[1:]) # Get game components based on input
    nn_model = generate_model()
    prng = Random()
    prng.seed(time())
    ea = ec.DEA(prng)
    ea.terminator = ec.terminators.evaluation_termination
    final_pop = ea.evolve(generator=generate_candidate,
                          evaluator=evaluate_candidates,
                          pop_size = 1000,
                          maximize = True,
                          max_evaluations = 1000,
                          mutation_rate = 0.3,
                          crossover_rate = 1,
                          nn_model = nn_model,
                          num_elites = 1,
                          mp_num_cpus=4,
                          cmd_line_args = cmd_line_args)
    best = max(final_pop)
    nn_model.set_weights(best.candidate)
    print("Final candidate:", best.candidate)
    pacmanType = pacman.loadAgent("BioAgent", True)
    cmd_line_args['pacman'] = pacmanType(nn_model = nn_model)
    cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
    cmd_line_args['numGames'] = 10
    games = pacman.runGames(**cmd_line_args)
    pass
